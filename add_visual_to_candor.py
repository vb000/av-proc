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
import time
from util import launch_array_jobs, chunked_iter
import json
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


def extract_visual_features_from_roi(video_path, models, task):
    MAX_VIDEO_LENGTH = 5 * 60 * 30
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
        seg_num = int(math.ceil(frames.shape[2] / MAX_VIDEO_LENGTH))
        features = []
        for si in range(seg_num):
            if (si + 1)*MAX_VIDEO_LENGTH > frames.shape[2]:
                end_i = frames.shape[2]
            else:
                end_i = (si + 1)*MAX_VIDEO_LENGTH

            segments = frames[:, :, si*MAX_VIDEO_LENGTH:end_i, ...]
            feature, _ = model.extract_finetune(
                source={'video': segments, 'audio': None}, padding_mask=None, output_layer=None)
            feature = feature.squeeze(dim=0)
            features.append(feature)
        features = torch.cat(features, dim = 0)

    return features



@torch.no_grad() 
def extract_avhubert(sample, out_folder, models, task, cnn_detector, predictor, video_config, device):
    
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

    sample_folder = sample['id']
    conv_id = os.path.basename(sample_folder)
    data_folder = os.path.join(sample_folder, "processed")
    speakers = sample['speaker_id']

    roi_path = os.path.join(out_folder, "roi")
    roi_folder = os.path.join(roi_path, conv_id)
    os.makedirs(roi_folder, exist_ok = True)
    visual_path = os.path.join(out_folder, "visual_feats")
    visual_folder = os.path.join(visual_path, conv_id)
    os.makedirs(visual_folder, exist_ok = True)

    offset = sample['start_offset']
    duration = sample['duration']
    FPS = 30
    video_config['start_offset'] = int(offset * FPS)

    feats_ids = []
    valid_ids = []

    for spk_id in speakers:
        video_id = os.path.join(data_folder, spk_id + ".mp4")
        roi_id = os.path.join(roi_folder, spk_id + ".mp4")
        feats_id = os.path.join(visual_folder, spk_id + ".npy")
        valid_id = os.path.join(visual_folder, spk_id + "_valid.npy")
        
        if not os.path.exists(roi_id) or not os.path.exists(valid_id):
            t0 = time.time()
            videogen = skvideo.io.vread(video_id)
            # videogen = videogen[start:]
            frames = np.array([frame for frame in videogen])

            frames = frames[video_config['start_offset']:, ...]
            print(f"loading video {video_id} done", offset, frames.shape, frames.shape[0]/FPS, duration/2)

            landmarks = []
            landmarks_valid = []
            for frame in frames:
                landmark = detect_landmark(frame, cnn_detector, predictor)
                if landmark is None:
                    landmarks_valid.append(False)
                else:
                    landmarks_valid.append(True)
                landmarks.append(landmark)
            preprocessed_landmarks = landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                return None, None

            rois = crop_patch(video_id, preprocessed_landmarks, **video_config)
            # print("rois- ", rois.shape)
            write_video_ffmpeg(rois, roi_id, FFMPEG)
            landmarks_valid = np.array(landmarks_valid)
            np.save(valid_id, landmarks_valid)
        else:
            landmarks_valid = np.load(valid_id)
        

        
        features = extract_visual_features_from_roi(roi_id, models, task)
        print("features - ", features.shape)
        features = features.cpu().numpy()
        np.save(feats_id, features)
        
        feats_ids.append(feats_id)
        valid_ids.append(valid_id)
    

    return feats_ids, valid_ids

def worker(
    in_file: str, out_file: str, resume: bool,
):
    assert out_file.endswith('.jsonl'), out_file
    log_file = out_file.rsplit('.', 1)[0] + '.log'
    err_file = out_file.rsplit('.', 1)[0] + '.err'
    print(f'Writing to {out_file=} {log_file=} {err_file=}...')

    mode = 'w'
    done_list = {}
    if os.path.exists(out_file) and resume:
        assert os.path.exists(log_file), log_file
        mode = 'a'
        with jsonlines.open(log_file, mode='r') as log_reader:
            for obj in log_reader:
                done_list[obj['id']] = True
    else:
        assert not (os.path.exists(out_file) or os.path.exists(log_file))

    #### loading model ####
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
        "start_offset": 0,
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

    out_folder = out_file.rsplit('/', 1)[0]

    with jsonlines.open(in_file, mode='r') as in_reader, \
            jsonlines.open(out_file, mode=mode, flush=True) as out_writer, \
            jsonlines.open(log_file, mode=mode, flush=True) as log_writer, \
            jsonlines.open(err_file, mode='w', flush=True) as err_writer:
        for data_i, sample in enumerate(in_reader): # iterate with each input file
            # infile ----> {"speaker_id": "xxx", "file": List[mp4_file] }
            print(f"processing {data_i}.....")        
            ## check the in file id
            id = sample['id']
            if id not in done_list:
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                    with Timer() as rt:
                        feats_id, landmarks_valid = extract_avhubert(sample, out_folder, models, task, cnn_detector, predictor, video_config, device)
                        if feats_id is None:
                            print("error!!!!! not roi detected!!!!!!")
                            continue
                        else:
                            sample['visual_path'] = feats_id
                            sample['valid_path'] = landmarks_valid
                            out_writer.write(sample)
                        
                    log_writer.write({
                        'id': id,
                        'runtime': rt.elapsed,
                        'max_cuda_mem (G)': torch.cuda.max_memory_allocated(device) / 1e9,
                    })
                except Exception as e:
                    err_writer.write({
                        'id': id,
                        'error': str(e),
                        'max_cuda_mem (G)': torch.cuda.max_memory_allocated(device) / 1e9,
                    })
            
@dataclass
class Config:
    data_dir: str
    out_dir: str
    num_jobs: int

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()

    config = Config(
        data_dir= "/fsx-ust/tuochao/dataset/candor_noise_dataset",
        out_dir='/fsx-ust/tuochao/dataset/candor_noise_av_dataset',
        num_jobs=64,
    )
    
    os.makedirs(config.out_dir, exist_ok = True)
    ## list all voxceleb2 data under the folder
    data_lists = [f for f in os.listdir(config.data_dir) if f.endswith(".jsonl")]
    print("total candor conversation folder ---- ", len(data_lists))
    num_jobs = len(data_lists)

    in_files = [os.path.join(config.data_dir, f) for f in data_lists] #list(chunked_iter(data_lists, max_chunks=num_jobs))
    out_files = [os.path.join(config.out_dir, f) for f in data_lists]
    resume = [True for _ in in_files]


    if args.launch:
        launch_array_jobs(
            config.out_dir,
            worker,
            [in_files, out_files, resume],
            job_name="candor",
            account='seamless',
            qos='seamless_high',
            time=36 * 60,
        )
    else:
        launch_array_jobs(
            'temp2',
            worker,
            [[_] for _ in ('/fsx-ust/tuochao/dataset/candor_noise_dataset/train.000.jsonl', 'temp2/temp.jsonl', True)],
            job_name="candor",
            account='seamless',
            qos='seamless_high'
        )