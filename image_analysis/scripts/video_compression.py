import time
import glob
import json
import os

import cv2
import numpy as np

import utils

VIDEO_SAVE_DIR = "/Users/connorparish/code/conbot/screencapture/screenshot_data/raw/"
VIDEO_CHANGES_DIR = f"/Users/connorparish/code/conbot/screencapture/video_changes/"

def get_change_box(frame_diff_mask, pixels_value_thresh=100):
    frame_diff_mask = cv2.medianBlur(frame_diff_mask, 5)
    x = 0
    while x < frame_diff_mask.shape[1] and np.count_nonzero(frame_diff_mask[:, x]) < pixels_value_thresh:
        x += 1
    y = 0
    while y < frame_diff_mask.shape[0] and np.count_nonzero(frame_diff_mask[y, :]) < pixels_value_thresh:
        y += 1
    r = frame_diff_mask.shape[1] - 1
    while r > 0 and np.count_nonzero(frame_diff_mask[:, r]) < pixels_value_thresh:
        r -= 1
    b = frame_diff_mask.shape[0] - 1
    while b > 0 and np.count_nonzero(frame_diff_mask[b, :]) < pixels_value_thresh:
        b -= 1

    # If no change box (area negative) is detected try again with a lower pixels_value_thresh
    if (r - x) * (b - y) < 0:
        return get_change_box(frame_diff_mask=frame_diff_mask, pixels_value_thresh=pixels_value_thresh/2)
    return x, y, r, b
    
def get_frame_change(frame_0_gray, frame_1_gray, diff_percentage_threshold=0.005, pixels_value_thresh=100):
    frame_diff = cv2.absdiff(frame_0_gray, frame_1_gray)
    frame_diff = frame_diff.astype(np.uint8)
    diff_percentage = np.count_nonzero(frame_diff) / frame_diff.size
    frame_diff_dict = {"diff_percent" : diff_percentage}
    if diff_percentage > diff_percentage_threshold: 
        frame_diff_mask = cv2.inRange(frame_diff, np.array([1]), np.array([255]))
        x, y, r, b = get_change_box(frame_diff_mask, pixels_value_thresh)
        frame_diff_dict['change_box'] = [x, y, r, b]
    return frame_diff_dict
    
def get_frame_changes(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_changes = list()
    prev_frame_gray = None
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break  # Reached the end of the video
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame_gray is not None:
            frame_changes.append(get_frame_change(prev_frame_gray, frame_gray))
        prev_frame_gray = frame_gray
    return frame_changes

def process_video_changes(video_path, save_path):
    print(f"Starting Processing for {video_path}")
    utils.make_dirs("/".join(save_path.split("/")[:-1]))
    start_time = time.time()
    frame_changes = get_frame_changes(video_path=video_path)
    if frame_changes is None:
        return

    with open(save_path, 'w') as outfile:
        json.dump(frame_changes, outfile)

    print(f"Time: {time.time() - start_time}. Wrote frame changes to: {save_path}")

def run_video_changes_processing(video_save_dir, video_change_save_dir):
    video_paths = glob.glob(f"{video_save_dir}*/*/*/*.mp4")
    for i, video_path in enumerate(video_paths):
        print(f"{i} out of {len(video_paths)}")
        vp = video_path.replace(video_save_dir, "")
        video_changes_file_name = vp.replace(".mp4", "_video_changes.json")
        video_changes_save_path = os.path.join(video_change_save_dir, video_changes_file_name)
        if os.path.exists(video_changes_save_path):
            continue
        process_video_changes(video_path=video_path, save_path=video_changes_save_path)

if __name__ == "__main__":
    run_video_changes_processing(VIDEO_SAVE_DIR, VIDEO_CHANGES_DIR)