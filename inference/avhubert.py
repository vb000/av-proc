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


def main(video_paths_chunk, out_file, work_dir):
    """Process a chunk of videos to extract transcriptions"""
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
    parser.add_argument('--video_dir', type=str,
                        default='/mmfs1/gscratch/intelligentsystems/common_datasets/VoxCeleb2/mp4/id00017/7t6lfzvVaTM',
                        help='Directory containing the video files to process')
    parser.add_argument('--out_file', type=str, default='./vsr_results.csv',
                        help='Path to save output CSV file')
    args = parser.parse_args()
    
    # Find all mp4 files in the specified directory
    video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    print(f"Found {len(video_paths)} video files in {args.video_dir}")

    if not video_paths:
        print(f"No video files found in {args.video_dir}")
        sys.exit(1)

    # Process the videos
    main(video_paths, args.out_file, work_dir)


# if __name__ == '__main__':
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description='Process videos for feature extraction and transcription')
#     parser.add_argument(
#         '--video_zip_file', type=str,
#         default="/mmfs1/gscratch/intelligentsystems/common_datasets/VoxCeleb2/vox2_test_mp4.zip",
#         help='Path to the directory containing video files'
#     )
#     parser.add_argument('--out_file', type=str, required=True, help='Path to save output CSV file')
#     args = parser.parse_args()

#     # Extract the zip file to /scr
#     print("Extracting zip file to /scr directory...")
#     os.makedirs("/scr", exist_ok=True)
#     extract_dir = "/scr/vox2_test_mp4"
#     os.makedirs(extract_dir, exist_ok=True)

#     # Check if files are already extracted
#     if not os.path.exists(os.path.join(extract_dir, "mp4")):
#         with zipfile.ZipFile(args.video_zip_file, 'r') as zip_ref:
#             zip_ref.extractall(extract_dir)
#         print("Extraction complete")
#     else:
#         print("Files already extracted, skipping extraction")

#     # Find all mp4 files in the extracted directory
#     print("Finding all mp4 files...")
#     video_paths = sorted(glob.glob(os.path.join(extract_dir, "**", "*.mp4"), recursive=True))
#     print(f"Found {len(video_paths)} video files")

#     # Create a dataframe with all video paths
#     all_videos_df = pd.DataFrame({'video_path': video_paths})

#     # Number of Slurm jobs
#     num_jobs = 128

#     # Calculate chunk size for approximately equal splits
#     chunk_size = math.ceil(len(video_paths) / num_jobs)

#     # Function to launch a Slurm job
#     def launch_slurm_job(job_id, video_chunk, base_out_file):
#         chunk_out_file = f"{os.path.splitext(base_out_file)[0]}_{job_id}{os.path.splitext(base_out_file)[1]}"
#         chunk_file = f"/tmp/video_chunk_{job_id}.txt"

#         # Save the chunk paths to a temporary file
#         with open(chunk_file, "w") as f:
#             for path in video_chunk:
#                 f.write(f"{path}\n")

#         # Create the Slurm job script
#         slurm_script  = f"#!/bin/bash\n"
#         slurm_script += f"#SBATCH --job-name=avhubert_{job_id}\n"
#         slurm_script += f"#SBATCH --output=avhubert_{job_id}_%j.out\n"
#         slurm_script += f"#SBATCH --error=avhubert_{job_id}_%j.err\n"
#         slurm_script += f"#SBATCH --partition=gpu-rtx6k\n"
#         slurm_script += f"#SBATCH --gres=gpu:1\n"
#         slurm_script += f"#SBATCH --cpus-per-task=4\n"
#         slurm_script += f"#SBATCH --mem=16G\n"
#         slurm_script += f"#SBATCH --time=24:00:00\n\n"
#         slurm_script += f"cd {os.getcwd()}\n"
#         slurm_script += f"python -c \"import sys, os\n"
#         slurm_script += f"sys.path.insert(0, '{os.getcwd()}')\n"
#         slurm_script += f"from inference.avhubert import main, work_dir\n\n"
#         slurm_script += f"# Load the video paths\n"
#         slurm_script += f"with open('{chunk_file}', 'r') as f:\n"
#         slurm_script += f"    video_paths = [line.strip() for line in f.readlines()]\n\n"
#         slurm_script += f"# Process the videos\n"
#         slurm_script += f"main(video_paths, '{chunk_out_file}', '{work_dir}')\n"
#         slurm_script += f"\""

#         # Write the Slurm script to a file
#         script_file = f"/tmp/slurm_job_{job_id}.sh"
#         with open(script_file, "w") as f:
#             f.write(slurm_script)

#         # Submit the job
#         subprocess.run(["sbatch", script_file])
#         print(f"Submitted job {job_id}")

#     # Launch Slurm jobs for each chunk
#     for job_id in range(num_jobs):
#         start_index = job_id * chunk_size
#         end_index = min(start_index + chunk_size, len(video_paths))
#         video_chunk = video_paths[start_index:end_index]

#         if not video_chunk:
#             print(f"No videos found for job {job_id}, skipping...")
#             continue

#         launch_slurm_job(job_id, video_chunk, args.out_file)
#     print("All Slurm jobs have been submitted.")
