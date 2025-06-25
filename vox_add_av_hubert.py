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

from util import launch_array_jobs

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
def extract_avhubert(video_id, models, task, cnn_detector, predictor, video_config, device):
    
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
    roi_path = roi_path.replace("mp4/", "mp4_roi/")
    roi_id = roi_path  +  "/" + video_name
    
    if not os.path.exists(roi_id):
        videogen = skvideo.io.vread(video_id)
        frames = np.array([frame for frame in videogen])
        print(f"loading video {video_id} done")

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

    return features 

@torch.no_grad()
def worker(
    data_jsonl: str, out_file: str, resume: bool
):
    assert out_file.endswith('.jsonl'), out_file
    log_file = out_file.rsplit('.', 1)[0] + '.log'
    err_file = out_file.rsplit('.', 1)[0] + '.err'
    print(f'Writing to {out_file=} {log_file=} {err_file=}...')
    pt_folder = os.path.join(out_file.rsplit('/', 1)[0], "visual_feats")
    os.makedirs(pt_folder, exist_ok=True)
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

    with jsonlines.open(data_jsonl, mode='r') as in_reader, \
            jsonlines.open(out_file, mode=mode, flush=True) as out_writer, \
            jsonlines.open(log_file, mode=mode, flush=True) as log_writer, \
            jsonlines.open(err_file, mode='w', flush=True) as err_writer:
        for n, item in enumerate(tqdm(in_reader)):
            ## for debug
            video_id = item['id']
            if not (os.path.exists(item['id']) and os.path.isfile(item['id'])):
                print(f"{item['id']} not existings!!!!!")
                continue

            if item['id'] not in done_list:
                try:
                    print(f"process.... {item['id']} at {n}")
                    torch.cuda.reset_peak_memory_stats(device)
                    with Timer() as rt:

                        visual_feats = extract_avhubert(item['id'], models, task, cnn_detector, predictor, video_config, device)
                        if visual_feats is None:
                            print(f"No face detected or very low quality of video input at {item['id']}")
                            continue
                        else:
                            print("features extract finished!!!")
                            foldler_name, _id, speaker_id, pt_id = item['id'].rsplit('/', 3)
                            # pt_id = (item['id'].rsplit('/', 1)[1]).rsplit('.', 1)[0]
                            visual_folder = os.path.join(pt_folder, _id, speaker_id)
                            os.makedirs(visual_folder, exist_ok = True)
                            visual_path = os.path.join(visual_folder, pt_id + ".npy") 
                            item['visual_path'] = visual_path
                            # torch.save(visual_feats, visual_path)
                            visual_feats = visual_feats.cpu().numpy()
                            np.save(visual_path, visual_feats)
                            out_writer.write(item)
                            
                    log_writer.write({
                        'id': item['id'],
                        'runtime': rt.elapsed,
                        'max_cuda_mem': torch.cuda.max_memory_allocated(device) / 1e9,
                    })
                    print(f"process.... {item['id']} succeed!!!!")
                except Exception as e:
                    print(f"process.... {item['id']} fail!!!!")
                    err_writer.write({
                        'id': item['id'],
                        'error': str(e),
                        'max_cuda_mem': torch.cuda.max_memory_allocated(device) / 1e9,
                    })
                    
@dataclass
class Config:
    in_dir: str
    out_dir: str
    num_jobs: int


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()

    # config = Config(
    #     in_dir= '/fsx-ust/tuochao/dataset/vox2_av/dev',
    #     out_dir= '/fsx-ust/tuochao/dataset/vox2_av_avhubert_large/dev',
    #     num_jobs=64,
    # )

    # config = Config(
    #     in_dir= '/fsx-ust/tuochao/dataset/vox2_av_avhubert/dev_en',
    #     out_dir= '/fsx-ust/tuochao/dataset/vox2_av_avhubert_large/dev_en',
    #     num_jobs=64,
    # )
    #prefix = "dev"

    config = Config(
        in_dir= '/fsx-ust/tuochao/dataset/vox2_av/test_en',
        out_dir= '/fsx-ust/tuochao/dataset/vox2_av_avhubert_large/test_en',
        num_jobs=64,
    )
    prefix = "test"



    num_jobs = config.num_jobs

    # prefix = os.path.basename(os.path.abspath(config.out_dir))
    
    in_files = [
        os.path.join(config.in_dir, f'{prefix}.{_:03d}.jsonl')
        for _ in range(num_jobs)
    ]

    out_files = [
        os.path.join(config.out_dir, f'{prefix}.{_:03d}.jsonl')
        for _ in range(num_jobs)
    ]
    resume = [True for _ in in_files]

    if args.launch:
        launch_array_jobs(
            config.out_dir,
            worker,
            [in_files, out_files, resume],
            job_name='vox_avhubert',
            account='seamless',
            qos='seamless_high',
            time=48 * 60,
        )
    else:
        launch_array_jobs(
            "/fsx-ust/tuochao/vb_syncllm/temp2",
            worker,
            [[_] for _ in ["/fsx-ust/tuochao/vb_syncllm/temp/temp.jsonl", "/fsx-ust/tuochao/vb_syncllm/temp2/temp.jsonl", resume] ],
            job_name='vox_avhubert',
            account='seamless',
            qos='seamless_high',
            time=24 * 60,
        )