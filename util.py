import typing as tp
from typing import Any, Dict, List, Optional, Tuple, Union, final, Callable
from pathlib import Path
import os
import io
import tempfile

import ffmpeg
import numpy as np
import torch
import torchaudio

import submitit
from skimage import transform as tf
from collections import deque

import warnings

def preprocess(model, frames, wav, video_fps, audio_sr, device):
    if hasattr(model, 'preprocess') and callable(model.preprocess):
        # for AVBERT
        processed_frames, processed_wav = model.preprocess(frames, wav, video_fps, audio_sr)
    else:
        if hasattr(model, 'preprocess_audio') and callable(model.preprocess_audio):
            processed_wav = model.preprocess_audio(wav, audio_sr)
        else:
            warnings.warn('Model does not implement preprocess_audio method.')
            processed_wav = wav
        if hasattr(model, 'preprocess_video') and callable(model.preprocess_video):
            processed_frames = model.preprocess_video(frames, video_fps)
        else:
            warnings.warn('Model does not implement preprocess_video method.')
            processed_frames = frames
    return processed_wav.to(device), processed_frames.to(device),

class AudioBytes(tp.NamedTuple):
    path: str
    byte_offset: int
    length: int

    def audio_segment_to_string(self) -> str:
        return ":".join((self.path, str(self.byte_offset), str(self.length)))

    def __str__(self) -> str:
        return self.audio_segment_to_string()

    def load(self, average_channels: bool = False) -> "torch.Tensor":
        with open(self.path, "rb") as f:
            f.seek(self.byte_offset)
            audio_bytes = f.read(self.length)

        if len(audio_bytes) != self.length:
            raise RuntimeError(
                f"Expected to read {self.length} bytes from {self.path} at offest"
                f" {self.byte_offset}, only read {len(audio_bytes)}"
            )

        wav, sample_rate = torchaudio.load(io.BytesIO(audio_bytes), backend='ffmpeg')
        if average_channels and wav.ndim > 1 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, sample_rate

def parse_audiozip_path(audio_path: str):
    path, byte_offset, length = audio_path.split(":")
    return AudioBytes(path, int(byte_offset), int(length))

def count_lines(filename):
    count = 0
    if os.path.exists(filename):
        for _ in open(filename): count += 1
    return count

def chunked_iter(l, max_chunks):
    """
    >>> for _ in chunked_iter([1, 2, 3, 4, 5], max_chunks=3): print(_)
    ...
    [1, 2]
    [3, 4]
    [5]
    >>> for _ in chunked_iter([1, 2, 3], max_chunks=5): print(_)
    ...
    [1]
    [2]
    [3]
    """
    chunk_sizes = [len(l) // max_chunks for _ in range(max_chunks)]
    for i in range(len(l) % max_chunks):
        chunk_sizes[i] += 1

    idx = 0
    for k in chunk_sizes:
        if k == 0:
            return
        yield l[idx : idx + k]
        idx += k

def launch_array_jobs(
    job_folder: Path,
    func: Callable,
    func_args,
    job_name: str,
    account: str = None,
    qos: str = None,
    partition: str = None,
    time: int = 60 * 48,
):
    os.makedirs(job_folder, exist_ok=True)

    log_root = os.path.join(job_folder, 'logs')
    os.makedirs(log_root, exist_ok=True)
    log_folder = f"{log_root}/%j"

    executor = submitit.SlurmExecutor(folder=log_folder)
    executor.update_parameters(
        time=time,
        account=account,
        qos=qos,
        partition=partition,
        nodes=1,
        ntasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=8,
        mem="0G",
        job_name=job_name,
    )

    jobs = executor.map_array(func, *func_args)


def mp4_to_wav(mp4_file):
    """
    Extracts the audio from an MP4 file and returns it as torch.Tensor.

    Args:
        mp4_file (str): The path to the MP4 file.

    Returns:
        tuple: A tuple containing the sample rate and the audio data.
    """
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
        input_stream = ffmpeg.input(mp4_file)
        audio_stream = input_stream.audio
        output_stream = ffmpeg.output(audio_stream, tmp_file.name, format='wav')
        ffmpeg.run(output_stream, overwrite_output=True)
        audio_data, sr = torchaudio.load(tmp_file.name)
        return audio_data, sr



# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped

# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):

    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

def crop_patch(frames_new, landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, window_margin, start_idx, stop_idx, crop_height, crop_width):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    num_frames = len(frames_new)
    margin = min(num_frames, window_margin)
    assert(len(landmarks) == num_frames)
    while True:
        if frame_idx >= num_frames:
            break
        frame = frames_new[frame_idx]
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                           mean_face_landmarks[stablePntsIDs, :],
                                           cur_frame,
                                           STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[start_idx:stop_idx],
                                        crop_height//2,
                                        crop_width//2,))
        if frame_idx == len(landmarks)-1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[start_idx:stop_idx],
                                            crop_height//2,
                                            crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None

# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks