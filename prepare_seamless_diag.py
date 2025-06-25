# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import typing as tp
from dataclasses import dataclass
import os
import glob
import math
from argparse import Namespace

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchaudio
import jsonlines
import dac
from audiotools import AudioSignal
from contexttimer import Timer
from skimage import transform as tf
import dlib
import cv2 
import skvideo
import skvideo.io
import sys
import git
import soundfile as sf
from util import launch_array_jobs, mp4_to_wav
import time 

FFMPEG = '/data/home/tuochao/miniconda3/bin/ffmpeg'
os.environ['PATH'] = f'{os.path.dirname(FFMPEG)}:' + os.environ['PATH']
root = git.Repo('.', search_parent_directories=True).working_tree_dir
work_dir = os.path.join(root, 'modules', 'AVHuBERT', 'avhubert')

# root = git.Repo('.', search_parent_directories=True).working_tree_dir
# work_dir = os.path.join(root, 'modules', 'AVHuBERT', 'avhubert')
# sys.path.insert(0, work_dir)
# print("*"*10, work_dir)

from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
import utils as avhubert_utils
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig


dataset_folder = "/fsx-ust/tuochao/raw/seamless_diag"
diag_id = ["B17_0189_00000027_0261", "B17_0189_00000027_0185"]
device = 'cuda'

video_config = {
    "STD_SIZE":  (256, 256),
    "mean_face_landmarks": np.load("/fsx-ust/tuochao/av-proc/data/misc/20words_mean_face.npy"),
    "stablePntsIDs": [33, 36, 39, 42, 45],
    "window_margin": 12,
    "start_idx": 48,
    "stop_idx" : 68,
    "crop_width" : 96,
    "crop_height" : 96,
}
face_predictor_path = "/fsx-ust/tuochao/av-proc/data/misc/shape_predictor_68_face_landmarks.dat" 
cnn_detector_path = "/fsx-ust/tuochao/av-proc/data/misc/mmod_human_face_detector.dat"
# ckpt_path = "/fsx-ust/tuochao/av-proc/data/checkpoints/base_vox_433h.pt"
ckpt_path = "/fsx-ust/tuochao/av-proc/data/checkpoints/self_large_vox_433h.pt"

## load vision model
cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
predictor = dlib.shape_predictor(face_predictor_path)
utils.import_user_module(Namespace(user_dir=work_dir))
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])



def extract_visual_features_from_roi(video_path, models, task):
    transform = avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])
    frames = avhubert_utils.load_video(video_path)
    frames = transform(frames)
    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    model = models[0]
    if hasattr(models[0], 'decoder'):
        model = models[0].encoder.w2v_model
    model.cuda()
    model.eval()
    with torch.no_grad():
        # Specify output_layer if you want to extract feature of an intermediate layer
        feature, _ = model.extract_finetune(
            source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None)
        feature = feature.squeeze(dim=0)
    return feature

@torch.no_grad() 
def extract_avhubert(video_id, transcript_id, models, task, cnn_detector, predictor, video_config, device):
    

    ### find all face in frame and return the largest one
    def detect_landmark(image, detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        dets = detector(gray, 1)
        rects = dlib.rectangles()
        rects.extend([d.rect for d in dets])

        coords = None
        if len(rects) < 1:
            return coords    
        ### if multiple faces appear and find the biggest one
        rects_size = []
        max_size = -9999
        max_i = -1
        for (i, rect) in enumerate(rects):
            rect_size = abs(rect.right() - rect.left()) * abs(rect.top() - rect.bottom())
            if rect_size > max_size:
                max_size = rect_size
                max_i = i 
        ## the large one
        rect = rects[max_i]
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    roi_path, video_name= video_id.rsplit('/', 1)
    roi_id = roi_path  +  "/" + "roi_" + video_name

    wav, SR = mp4_to_wav(video_id)
    wav = wav.mean(0, keepdim=True) / torch.max(torch.abs(wav)) # -1 - 1
    
    if not os.path.exists(roi_id):
        videogen = skvideo.io.vread(video_id)
        frames = np.array([frame for frame in videogen])
        print(f"loading video {video_id} done", frames.shape)

        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, cnn_detector, predictor)
            landmarks.append(landmark)
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        if preprocessed_landmarks is None:
            return None

        rois = crop_patch(video_id, preprocessed_landmarks, **video_config)
        write_video_ffmpeg(rois, roi_id, FFMPEG)

    features = extract_visual_features_from_roi(roi_id, models, task)
    print("features", features.shape)
    FPS = 30
    # np.save(feat_id, features)
    
    seg_i = 0
    samples = []
    with jsonlines.open(transcript_id, mode='r') as in_reader:
        for line in in_reader:
            if line['end'] - line['start'] < 2:
                continue
            start_idx = int(line['start'] * FPS)
            end_idx = int(line['end'] * FPS)
            visual = features[start_idx:end_idx, :]

            
            start_idx_audio = int(line['start'] * SR)
            end_idx_audio = int(line['end'] * SR)
            wav = wav[:, start_idx_audio:end_idx_audio]
            wav_file = os.path.join(dataset_folder, "out", f"seg_{seg_i}.wav")
            sf.write(wav_file, wav.T, SR)

            duration = int(visual.shape[0] / 30 *25)
            feat_id = roi_path + "/visual/" + video_name + f"_{seg_i}.npy"
            seg_i = seg_i + 1
            np.save(feat_id, visual.cpu().numpy())


            ## subtract offset
            start_offset = line['start'] 
            for i in range(len(line["words"])):
                if 'start' in line["words"][i].keys():
                    line["words"][i]['start'] -= start_offset

                if 'end' in line["words"][i].keys():
                    line["words"][i]['end'] -= start_offset

            sample = {
                'id': video_id,
                'duration': duration,
                'visual_path': feat_id,
                'transcript': line,
                'audio_path': wav_file,
            }
            samples.append(sample)

    return samples 


for _id in diag_id:
    video_id = os.path.join(dataset_folder, _id + ".mp4")
    transcript_id = os.path.join(dataset_folder, "text", _id + ".jsonl")
    print(video_id)
    samples = extract_avhubert(video_id, transcript_id, models, task, cnn_detector, predictor, video_config, device)

    out_file = os.path.join(dataset_folder, "out",  "temp.jsonl")
    with jsonlines.open(out_file, mode='w', flush=True) as out_writer:
        for _sample in samples:
            out_writer.write(_sample)
    # t0 = time.time()
    # clip = VideoFileClip(video_id)
    # video_array = np.array(list(clip.iter_frames()))
    # print("loading moviepy sub_clip", time.time() - t0)

    # t0 = time.time()
    # video = mediapy.read_video(video_id)
    # print("loading mediapy sub_clip", time.time() - t0)

    break
    # t0 = time.time()
    # av = de.AVReader(video_id, ctx, sample_rate=16000) # We take 22050 sample rate here just for demonstration
    # start_time = 0
    # end_time = 4 * 60 * 30
    # audio, video = av[0:7200]
    # print("loading sub_clip", time.time() - t0)
    # print('Frame #: ', len(audio))
    # # print('Shape of the first frame: ', video.asnumpy().shape)

    # break
    # t0 = time.time()
    # clip = VideoFileClip(video_id)
    # start_time = 0
    # end_time = 4 * 60  # seconds
    # # Extract the sub-clip
    # sub_clip = clip.subclip(start_time, end_time)
    # print("loading sub_clip", time.time() - t0)
    # t0 = time.time()
    # video_array = np.array(list(sub_clip.iter_frames()))
    # print(time.time() - t0, video_array.shape)

    


