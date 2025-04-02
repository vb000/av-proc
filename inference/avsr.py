import os
import glob
import argparse
from argparse import Namespace
import math

import pandas as pd
from tqdm import tqdm
import submitit
import subprocess


def main(video_zip, rank, world_size, out_dir, num_samples=None):
    """Process a chunk of videos to extract transcriptions"""

    from inference.avhubert import (
        root, work_dir, utils, checkpoint_utils, visual_speech_recognition
    )

    out_file = os.path.join(out_dir, f"vsr_results_rank{rank:03d}.csv")

    # Use subprocess to unzip video_zip to /scr directory
    print(f"Extracting zip file {video_zip} to /scr directory...")
    extract_dir = f"/scr/vox2_test_mp4/{rank:03d}"
    if os.path.exists(extract_dir):
        subprocess.run(["rm", "-rf", extract_dir], check=True)  # Remove existing directory
    os.makedirs(extract_dir, exist_ok=True)
    subprocess.run(["unzip", "-q", video_zip, "-d", extract_dir], check=True)
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
    parser.add_argument('--num_jobs', type=int, default=64,
                        help='Number of Slurm jobs to run in parallel')
    parser.add_argument('--partition', type=str, default='ckpt',
                        help='Slurm partition to use')
    parser.add_argument('--account', type=str, default='intelligentsystems',
                        help='Slurm account to use')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of video samples to process per job (None for all)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Create logs directory inside out_dir
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize executor with logs directory
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_gres="gpu:1",
        slurm_account=args.account,
        slurm_ntasks_per_node=1,
        slurm_time="24:00:00",
        slurm_mem="32G",
        slurm_constraint="rtx6k",
        name="avhubert_vsr",
        nodes=1,
    )

    video_zip = [args.video_zip_dir for _ in range(args.num_jobs)]  # Replicate the video_zip path for each job
    rank = [ i for i in range(args.num_jobs)]  # Create a list of ranks from 0 to num_jobs-1
    world_size = [args.num_jobs for _ in range(args.num_jobs)]  # All jobs have the same world size
    out_dir = [args.out_dir for _ in range(args.num_jobs)]  # Replicate the out_dir path for each job
    num_samples = [args.num_samples for _ in range(args.num_jobs)]  # Set to None to process all videos, or specify a limit
    executor.map_array(
        main,
        video_zip,
        rank,
        world_size,
        out_dir,
        num_samples
    )
    print("All jobs have been submitted.")
