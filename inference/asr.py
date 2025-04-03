import os
import glob
import argparse
from argparse import Namespace
import math

import pandas as pd
from tqdm import tqdm
import submitit
import subprocess
import whisperx


def get_audio_chunk(audio_zip, rank, world_size, num_samples=None):
    """Extract audio files from zip and return the chunk for this rank"""
    # Use subprocess to unzip audio_zip to /scr directory
    print(f"Extracting zip file {audio_zip} to /scr directory...")
    extract_dir = f"/scr/audio_files/{rank:03d}"
    if os.path.exists(extract_dir):
        subprocess.run(["rm", "-rf", extract_dir], check=True)  # Remove existing directory
    os.makedirs(extract_dir, exist_ok=True)
    subprocess.run(["unzip", "-q", audio_zip, "-d", extract_dir], check=True)
    print("Extraction complete")

    # Find all m4a files in the extracted directory
    print("Finding all m4a files...")
    audio_paths = sorted(glob.glob(os.path.join(extract_dir, "**", "*.m4a"), recursive=True))
    if not audio_paths:
        print(f"No audio files found in {extract_dir}")
        return []
    print(f"Found {len(audio_paths)} audio files in {extract_dir}")

    # Split audio paths into chunks for processing
    num_audios = len(audio_paths)
    chunk_size = math.ceil(num_audios / world_size)
    start_index = rank * chunk_size
    end_index = min(start_index + chunk_size, num_audios)

    if start_index >= num_audios:
        print(f"No audio files to process for rank {rank}, exiting...")
        return []

    audio_paths_chunk = audio_paths[start_index:end_index]
    if not audio_paths_chunk:
        print(f"No audio files found for rank {rank}, exiting...")
        return []

    print(f"Processing {len(audio_paths_chunk)} audio files for rank {rank}...")

    # Limit samples if specified
    if num_samples is not None:
        audio_paths_chunk = audio_paths_chunk[:num_samples]

    return audio_paths_chunk


def transcribe_audio_files(audio_paths, out_file):
    """Process a list of audio files to extract transcriptions using WhisperX"""
    # Load WhisperX model
    print("Loading WhisperX model...")
    device = "cuda"
    model = whisperx.load_model("large-v2", device)

    # Check for existing output and load already processed audios
    processed_audios = set()
    if os.path.exists(out_file):
        try:
            existing_df = pd.read_csv(out_file)
            processed_audios = set(existing_df['audio_path'].tolist())
            print(f"Found {len(processed_audios)} already processed audio files")
        except Exception as e:
            print(f"Error reading existing output file: {e}")

    # Create output file if it doesn't exist
    if not os.path.exists(out_file):
        pd.DataFrame(columns=['audio_path', 'asr_text']).to_csv(out_file, index=False)

    # Process each audio file and extract transcriptions
    for i, audio_path in enumerate(tqdm(audio_paths)):
        if audio_path in processed_audios:
            print(f"Skipping already processed audio: {audio_path}")
            continue
        if not os.path.isfile(audio_path):
            print(f"File not found, skipping: {audio_path}")
            continue

        # Ensure the audio path is valid
        audio_path = os.path.abspath(audio_path)
        try:
            # Use WhisperX to transcribe audio
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio)
            transcription = ' '.join([x['text'] for x in result["segments"]])

            # Save the result to the output file
            new_entry = pd.DataFrame(
                {'path': [audio_path], 'text': [transcription], 'language': [result['language']]}
            )
            new_entry.to_csv(out_file, mode='a', header=False, index=False)
            print(f"Processed audio {i + 1}/{len(audio_paths)}: {audio_path} -> {transcription}")
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")

    print(f"Completed processing {len(audio_paths)} audio files. Results saved to {out_file}.")


def main(audio_zip, rank, world_size, out_dir, num_samples=None):
    """Process a chunk of audio files to extract transcriptions using WhisperX"""
    out_file = os.path.join(out_dir, f"asr_results_rank{rank:03d}.csv")

    # Get the chunk of audio files for this rank
    audio_paths_chunk = get_audio_chunk(audio_zip, rank, world_size, num_samples)

    # Exit if no files to process
    if not audio_paths_chunk:
        return

    # Process the audio files
    transcribe_audio_files(audio_paths_chunk, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio files for transcription using WhisperX')
    parser.add_argument('--audio_zip_dir', type=str,
                        default='/mmfs1/gscratch/intelligentsystems/common_datasets/VoxCeleb2/vox2_test_aac.zip',
                        help='Directory containing the audio files to process')
    parser.add_argument('--out_dir', type=str, default='data/asr_outputs',
                        help='Path to save output CSV file')
    parser.add_argument('--num_jobs', type=int, default=64,
                        help='Number of Slurm jobs to run in parallel')
    parser.add_argument('--partition', type=str, default='ckpt',
                        help='Slurm partition to use')
    parser.add_argument('--account', type=str, default='intelligentsystems',
                        help='Slurm account to use')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of audio samples to process per job (None for all)')
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
        cpus_per_task=4,  # Number of CPU cores per task
        name="whisperx_asr",
        nodes=1,
    )

    audio_zip = [args.audio_zip_dir for _ in range(args.num_jobs)]  # Replicate the audio_zip path for each job
    rank = [i for i in range(args.num_jobs)]  # Create a list of ranks from 0 to num_jobs-1
    world_size = [args.num_jobs for _ in range(args.num_jobs)]  # All jobs have the same world size
    out_dir = [args.out_dir for _ in range(args.num_jobs)]  # Replicate the out_dir path for each job
    num_samples = [args.num_samples for _ in range(args.num_jobs)]  # Set to None to process all audios, or specify a limit
    executor.map_array(
        main,
        audio_zip,
        rank,
        world_size,
        out_dir,
        num_samples
    )
    print("All jobs have been submitted.")
