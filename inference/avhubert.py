import os, sys, shutil
from base64 import b64encode
import glob
import argparse
from argparse import Namespace
import tempfile
import zipfile
import subprocess
import math

import git
import pandas as pd
import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
import torch

FFMPEG = '/mmfs1/gscratch/cse/bandhav/miniconda3/envs/avhubert_gpu/bin/ffmpeg'
os.environ['PATH'] = f'{os.path.dirname(FFMPEG)}:' + os.environ['PATH']

root = git.Repo('.', search_parent_directories=True).working_tree_dir
work_dir = os.path.join(root, 'modules', 'AVHuBERT', 'avhubert')
sys.path.insert(0, work_dir)

from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
import utils as avhubert_utils
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig


def detect_landmark(image, detector, predictor, cnn_detector=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if cnn_detector:
        dets = detector(gray, 1)
        rects = dlib.rectangles()
        rects.extend([d.rect for d in dets])
    else:
        rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path,
                     cnn_detector_path=None):
    if cnn_detector_path is not None:
        detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    else:
        detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    videogen = skvideo.io.vread(input_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []
    for frame in frames:
        landmark = detect_landmark(frame, detector, predictor, cnn_detector=cnn_detector_path is not None)
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
    origin_clip_path, face_predictor_path, mean_face_path, models, task, cnn_detector_path=None
):
    # Create a temporary file for mouth_roi_path
    with tempfile.NamedTemporaryFile(suffix='.mp4') as mouth_roi:
        mouth_roi_path = mouth_roi.name

        # Call the preprocess_video function
        preprocess_video(
            origin_clip_path, mouth_roi_path, face_predictor_path, mean_face_path,
            cnn_detector_path=cnn_detector_path)

        # Call the feature extraction function
        feature = extract_visual_features_from_roi(mouth_roi_path, models, task)

    # Return the feature variable
    return feature


def visual_speech_recognition_from_roi(video_path, models, saved_cfg, task):
    num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    data_dir = tempfile.mkdtemp()
    tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
    label_cont = ["DUMMY\n"]
    with open(f"{data_dir}/test.tsv", "w") as fo:
        fo.write("".join(tsv_cont))
    with open(f"{data_dir}/test.wrd", "w") as fo:
        fo.write("".join(label_cont))

    modalities = ["video"]
    gen_subset = "test"
    gen_cfg = GenerationConfig(beam=20)
    models = [model.eval().cuda() for model in models]

    saved_cfg.task.modalities = modalities
    saved_cfg.task.data = data_dir
    saved_cfg.task.label_dir = data_dir
    task = tasks.setup_task(saved_cfg.task)
    task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
    generator = task.build_generator(models, gen_cfg)

    def decode_fn(x):
        dictionary = task.target_dictionary
        symbols_ignore = generator.symbols_to_strip_from_output
        symbols_ignore.add(dictionary.pad())
        return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

    itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
    sample = next(itr)
    sample = utils.move_to_cuda(sample)
    hypos = task.inference_step(generator, models, sample)
    ref = decode_fn(sample['target'][0].int().cpu())
    hypo = hypos[0][0]['tokens'].int().cpu()
    hypo = decode_fn(hypo)

    return hypo


def visual_speech_recognition(
    video_path, face_predictor_path, mean_face_path, models, saved_cfg, task, cnn_detector_path=None
):
    with tempfile.NamedTemporaryFile(suffix='.mp4') as mouth_roi:
        mouth_roi_path = mouth_roi.name

        # Call the preprocess_video function
        preprocess_video(
            video_path, mouth_roi_path, face_predictor_path, mean_face_path,
            cnn_detector_path=cnn_detector_path
        )

        hypo = visual_speech_recognition_from_roi(mouth_roi_path, models, saved_cfg, task)

    return hypo


def main(video_zip, rank, world_size, out_dir, work_dir, num_samples=None):
    """Process a chunk of videos to extract transcriptions"""

    out_file = os.path.join(out_dir, f"vsr_results_rank{rank:03d}.csv")

    # Unzip `video_zip` to `/scr` directory
    print(f"Extracting zip file {video_zip} to /scr directory...")
    extract_dir = "/scr/vox2_test_mp4"
    if os.path.exists(extract_dir):
        os.rmdir(extract_dir)  # Remove existing directory to avoid conflicts
    os.makedirs(extract_dir)
    with zipfile.ZipFile(video_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete")

    # Find all mp4 files in the extracted directory
    print("Finding all mp4 files...")
    video_paths = sorted(glob.glob(os.path.join(extract_dir, "**", "*.mp4"), recursive=True))
    if not video_paths:
        print(f"No video files found in {extract_dir}")
        return
    print(f"Found {len(video_paths)} video files in {extract_dir}")

    # Split video paths into chunks for processing
    num_videos = len(video_paths)
    num_chunks = world_size
    chunk_size = math.ceil(num_videos / num_chunks)
    start_index = rank * chunk_size
    end_index = min(start_index + chunk_size, num_videos)
    if start_index >= num_videos:
        print(f"No videos to process for rank {rank}, exiting...")
        return
    video_paths_chunk = video_paths[start_index:end_index]
    if not video_paths_chunk:
        print(f"No videos found for rank {rank}, exiting...")
        return
    print(f"Processing {len(video_paths_chunk)} videos for rank {rank}...")

    face_predictor_path = f"{root}/data/misc/shape_predictor_68_face_landmarks.dat"
    mean_face_path = f"{root}/data/misc/20words_mean_face.npy"
    ckpt_path = f"{root}/data/checkpoints/base_vox_433h.pt"
    cnn_detector_path = f'{root}/data/misc/mmod_human_face_detector.dat'

    utils.import_user_module(Namespace(user_dir=work_dir))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

    # Check for existing output and load already processed videos
    processed_videos = set()
    if os.path.exists(out_file):
        try:
            existing_df = pd.read_csv(out_file)
            processed_videos = set(existing_df['video_path'].tolist())
            print(f"Found {len(processed_videos)} already processed videos")
        except Exception as e:
            print(f"Error reading existing output file: {e}")

    # Create output file if it doesn't exist
    if not os.path.exists(out_file):
        pd.DataFrame(columns=['video_path', 'vsr_text']).to_csv(out_file, index=False)

    # Process each video and extract transcriptions
    for i, video_path in enumerate(tqdm(video_paths_chunk)):
        if num_samples is not None and i >= num_samples:
            print(f"Reached sample limit of {num_samples}, stopping processing.")
            break

        if video_path in processed_videos:
            print(f"Skipping already processed video: {video_path}")
            continue
        if not os.path.isfile(video_path): 
            print(f"File not found, skipping: {video_path}")
            continue
        # Ensure the video path is valid
        video_path = os.path.abspath(video_path) 
        try:
            # Use the visual_speech_recognition function to get the transcription
            transcription = visual_speech_recognition(
                video_path,
                face_predictor_path,
                mean_face_path,
                models,
                saved_cfg,
                task,
                cnn_detector_path=cnn_detector_path
            )

            # Save the result to the output file
            new_entry = pd.DataFrame({'video_path': [video_path], 'vsr_text': [transcription]})
            new_entry.to_csv(out_file, mode='a', header=False, index=False)
            print(f"Processed video {i + 1}/{len(video_paths_chunk)}: {video_path} -> {transcription}")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
    print(f"Completed processing {len(video_paths_chunk)} videos. Results saved to {out_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos for feature extraction and transcription')
    parser.add_argument('--video_zip_dir', type=str,
                        default='/mmfs1/gscratch/intelligentsystems/common_datasets/VoxCeleb2/vox2_test_mp4.zip',
                        help='Directory containing the video files to process')
    parser.add_argument('--out_dir', type=str, default='data/vsr_outputs',
                        help='Path to save output CSV file')
    args = parser.parse_args()

    os.makedirs(args.out_dir)

    # Launch 64 batch jobs using submitit
    import submitit

    num_jobs = 64  # Number of Slurm jobs
    executor = submitit.AutoExecutor(folder=args.out_dir)
    executor.update_parameters(
        slurm_partition="gpu-rtx6k",
        slurm_gres="gpu:1",
        slurm_time="24:00:00",
        slurm_mem="16G",
        slurm_constraint="rtx6k",
        job_name="avhubert_vsr",
        nodes=1,
        tasks_per_node=1,
    )

    video_zip = [args.video_zip_dir for _ in range(num_jobs)]  # Replicate the video_zip path for each job
    rank = [ i for i in range(num_jobs)]  # Create a list of ranks from 0 to num_jobs-1
    world_size = [num_jobs for _ in range(num_jobs)]  # All jobs have the same world size
    out_dir = [args.out_dir for _ in range(num_jobs)]  # Replicate the out_dir path for each job
    work_dir = [work_dir for _ in range(num_jobs)]  # Replicate the work_dir path for each job
    executor.map_array(
        main,
        video_zip,
        rank,
        world_size,
        out_dir,
        work_dir,
        num_samples=None  # Set to None to process all videos
    )
    print("All jobs have been submitted.")
