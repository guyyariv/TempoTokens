"""
AV-Align Metric: Audio-Video Alignment Evaluation

AV-Align is a metric for evaluating the alignment between audio and video modalities in multimedia data.
It assesses synchronization by detecting audio and video peaks and calculating their Intersection over Union (IoU).
A higher IoU score indicates better alignment.

Usage:
- Provide a folder of video files as input.
- The script calculates the AV-Align score for the set of videos.
"""


import argparse
import glob
import cv2
import librosa
import librosa.display


# Function to extract frames from a video file
def extract_frames(video_path):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        frames (list): List of frames extracted from the video.
        frame_rate (float): Frame rate of the video.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise ValueError("Error: Unable to open the video file.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_rate


# Function to detect audio peaks using the Onset Detection algorithm
def detect_audio_peaks(audio_file):
    """
    Detect audio peaks using the Onset Detection algorithm.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        onset_times (list): List of times (in seconds) where audio peaks occur.
    """
    y, sr = librosa.load(audio_file)
    # Calculate the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Get the onset events
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


# Function to find local maxima in a list
def find_local_max_indexes(arr, fps):
    """
    Find local maxima in a list.

    Note:
        In this implementation, local maxima with an optical flow magnitude less than 0.1
        are ignored. This has always been the case to prevent static scenes from being
        incorrectly calculated as peaks due to very small optical flow.

    Args:
        arr (list): List of values to find local maxima in.
        fps (float): Frames per second, used to convert indexes to time.

    Returns:
        local_extrema_indexes (list): List of times (in seconds) where local maxima occur.
    """
    local_extrema_indexes = []
    n = len(arr)
    for i in range(1, n - 1):
        # Only consider local maxima with magnitude at least 0.1
        if arr[i - 1] < arr[i] > arr[i + 1] and arr[i] >= 0.1:
            local_extrema_indexes.append(i / fps)
    return local_extrema_indexes


# Function to detect video peaks using Optical Flow
def detect_video_peaks(frames, fps):
    """
    Detect video peaks using Optical Flow.

    Args:
        frames (list): List of video frames.
        fps (float): Frame rate of the video.

    Returns:
        flow_trajectory (list): List of optical flow magnitudes for each frame.
        video_peaks (list): List of times (in seconds) where video peaks occur.
    """
    flow_trajectory = [compute_of(frames[0], frames[1])] + [compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    video_peaks = find_local_max_indexes(flow_trajectory, fps)
    return flow_trajectory, video_peaks


# Function to compute the optical flow magnitude between two frames
def compute_of(img1, img2):
    """
    Compute the optical flow magnitude between two video frames.

    Args:
        img1 (numpy.ndarray): First video frame.
        img2 (numpy.ndarray): Second video frame.

    Returns:
        avg_magnitude (float): Average optical flow magnitude for the frame pair.
    """
    # Calculate the optical flow
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of the optical flow vectors
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    avg_magnitude = cv2.mean(magnitude)[0]
    return avg_magnitude


# Function to calculate Intersection over Union (IoU) for audio and video peaks
def calc_intersection_over_union(audio_peaks, video_peaks, fps):
    """
    Calculate Intersection over Union (IoU) between audio and video peaks.

    Note:
        A video peak is matched to at most one audio peak, as has always been the case,
        ensuring that a single video peak does not correspond to multiple audio peaks.

    Args:
        audio_peaks (list): List of audio peak times (in seconds).
        video_peaks (list): List of video peak times (in seconds).
        fps (float): Frame rate of the video.

    Returns:
        iou (float): Intersection over Union score.
    """
    intersection_length = 0
    used_video_peaks = [False] * len(video_peaks)
    for audio_peak in audio_peaks:
        for j, video_peak in enumerate(video_peaks):
            if not used_video_peaks[j] and video_peak - 1 / fps < audio_peak < video_peak + 1 / fps:
                intersection_length += 1
                used_video_peaks[j] = True
                break
    return intersection_length / (len(audio_peaks) + len(video_peaks) - intersection_length)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help='Insert the videos folder path')
    args = parser.parse_args()

    files = [file for file in glob.glob(args.input_dir + '*.mp4')]
    score = 0

    for file in files:

        file = file[:-4]
        video_path = f'{file}.mp4'
        audio_path = f'{file}.wav'

        frames, fps = extract_frames(video_path)

        audio_peaks = detect_audio_peaks(audio_path)
        flow_trajectory, video_peaks = detect_video_peaks(frames, fps)

        score += calc_intersection_over_union(audio_peaks, video_peaks, fps)

    print('AV-Align: ', score/len(files))
