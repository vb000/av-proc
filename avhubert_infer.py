import os, sys, shutil
import git

from base64 import b64encode
import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
import tempfile
import torch
from argparse import Namespace

FFMPEG = '/mmfs1/gscratch/cse/bandhav/miniconda3/envs/avhubert/bin/ffmpeg'
os.environ['PATH'] = f'{os.path.dirname(FFMPEG)}:' + os.environ['PATH']

root = git.Repo('.', search_parent_directories=True).working_tree_dir
work_dir = os.path.join(root, 'modules', 'AVHuBERT', 'avhubert')
sys.path.insert(0, work_dir)

from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
import utils as avhubert_utils
from fairseq import checkpoint_utils, options, tasks, utils


def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    videogen = skvideo.io.vread(input_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []
    for frame in tqdm(frames):
        landmark = detect_landmark(frame, detector, predictor)
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    write_video_ffmpeg(rois, output_video_path, FFMPEG)
    return


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


def extract_visual_features_from_video(
    origin_clip_path, face_predictor_path, mean_face_path, models, task
):
    # Create a temporary file for mouth_roi_path
    with tempfile.NamedTemporaryFile(suffix='.mp4') as mouth_roi:
        mouth_roi_path = mouth_roi.name

        # Call the preprocess_video function
        preprocess_video(origin_clip_path, mouth_roi_path, face_predictor_path, mean_face_path)

        # Call the feature extraction function
        feature = extract_visual_features_from_roi(mouth_roi_path, models, task)

    # Return the feature variable
    return feature


if __name__ == '__main__':
    face_predictor_path = f"{root}/data/misc/shape_predictor_68_face_landmarks.dat"
    mean_face_path = f"{root}/data/misc/20words_mean_face.npy"
    ckpt_path = f"{root}/data/checkpoints/base_vox_433h.pt"

    utils.import_user_module(Namespace(user_dir=work_dir))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

    features = extract_visual_features_from_roi(f'roi.mp4', models, task)
    print(features.shape) # [seq_len, 768]
