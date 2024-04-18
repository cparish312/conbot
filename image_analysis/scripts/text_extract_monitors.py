"""Script that uses ocrmac to extract text from monitors within screenshot videos."""

import os
import glob
import json
import multiprocessing
from collections import defaultdict
import cv2
from PIL import Image

from ocrmac import ocrmac

import utils

VIDEO_SAVE_DIR = "/Users/connorparish/code/conbot/screencapture/screenshot_data/raw/"
VIDEO_CHANGES_DIR = "/Users/connorparish/code/conbot/screencapture/video_changes/"
TEXT_EXTRACT_DIR = "/Users/connorparish/code/conbot/screencapture/video_text_extract_ocrmac/"

class VideoTextExtractor:
    def __init__(self, video_path, text_extract_save_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        self.open_video_metadata(video_path)
        if self.metadata is None:
            raise ValueError(f"No video metadata for: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_changes = self.load_frame_changes(video_path)
        if self.frame_changes is None:
            raise ValueError(f"No frame changes metadata for: {video_path}")

        print(f"{video_path} Total frames: {self.total_frames} Monitors: {len(self.metadata['monitors'])}")
        self.extract_text(text_extract_save_path)
            
    def load_frame_changes(self, video_path):
        vp = video_path.replace(VIDEO_SAVE_DIR, "")
        video_changes_file_name = vp.replace(".mp4", "_video_changes.json")
        video_changes_save_path = os.path.join(VIDEO_CHANGES_DIR, video_changes_file_name)
        if not os.path.exists(video_changes_save_path):
            return None
        
        with open(video_changes_save_path, 'r') as infile:
            return json.load(infile)
        
    def open_video_metadata(self, video_path):
        self.metadata = None
        metadata_path = video_path.replace("connor_mac_screen_recording", "connor_mac_screen_recording_metadata").replace("mp4", "json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as infile:
                metadata_d = json.load(infile)
            self.metadata = metadata_d

    def extract_text_from_frame(self, frame):
        ocr_res = ocrmac.OCR(Image.fromarray(frame), recognition_level='accurate').recognize(px=True) # px converts to pil coordinates
        return ocr_res
    
    def extract_text_from_monitor(self, monitor, frame):
        monitor_frame = frame[monitor['top']:monitor['top'] + monitor['height'], monitor['left']:monitor['left'] + monitor['width']]
        monitor_text = self.extract_text_from_frame(monitor_frame)
        return monitor_text
    
    def tuple_to_str(self, tup):
        s = "("
        for v in tup:
            s += str(v) + ","
        return s[:-1] + ")"

    def extract_text(self, text_extract_save_path):
        self.frame_num_to_extracted_text = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        frame_num = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        for i, monitor in enumerate(self.metadata['monitors']):
            self.frame_num_to_extracted_text[i][frame_num]['brute'][self.tuple_to_str((monitor['left'], monitor['top'], monitor['width'], monitor['height']))] = self.extract_text_from_monitor(monitor, frame)

        while True:
            # Read the next frame
            ret, frame = self.cap.read()
            frame_num += 1
            if not ret:
                break  # Reached the end of the video

            if self.frame_changes[frame_num - 1]['diff_percent'] < 0.005:
                continue
            def get_monitor_cb_overlap(cb, monitor):
                x1, y1, r1, b1 = cb

                # Determine the (x, y) coordinates of the overlap rectangle's bottom-left corner
                overlap_x1 = max(x1, monitor['left']) 
                overlap_y1 = max(y1, monitor['top']) 
                
                # Determine the (x, y) coordinates of the overlap rectangle's top-right corner
                overlap_x2 = min(r1, monitor['left'] + monitor['width']) 
                overlap_y2 = min(b1, monitor['top'] + monitor['height']) 
                
                # Calculate the dimensions of the overlap rectangle
                overlap_width = max(0, overlap_x2 - overlap_x1)
                overlap_height = max(0, overlap_y2 - overlap_y1)

                # Compute the area of the overlap rectangle
                overlap_area = overlap_width * overlap_height
                if overlap_area == 0:
                    return None
                return (overlap_x1, overlap_y1, overlap_width, overlap_height)
            
            cb = self.frame_changes[frame_num - 1]['change_box']
            for i, monitor in enumerate(self.metadata['monitors']):
                cb_monitor_overlap = get_monitor_cb_overlap(cb, monitor)
                if cb_monitor_overlap is not None and cb_monitor_overlap[2] > 5 and cb_monitor_overlap[3] > 5:
                    print(frame_num, i, cb_monitor_overlap)
                    cb_monitor_overlap_frame = frame[cb_monitor_overlap[1]:cb_monitor_overlap[1] + cb_monitor_overlap[3], cb_monitor_overlap[0]: cb_monitor_overlap[0]+cb_monitor_overlap[2]]
                    self.frame_num_to_extracted_text[i][frame_num]['brute'][self.tuple_to_str(cb_monitor_overlap)] = self.extract_text_from_frame(cb_monitor_overlap_frame)

        utils.make_dirs("/".join(text_extract_save_path.split("/")[:-1]))
        with open(text_extract_save_path, 'w') as outfile:
            json.dump(self.frame_num_to_extracted_text, outfile)

def run_text_extract(video_path, text_extract_save_path):
    try:
        VideoTextExtractor(video_path=video_path, text_extract_save_path=text_extract_save_path)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)

def run_text_extract_multiprocessing(video_save_dir, cpu_count=os.cpu_count()):
    video_paths_to_run = list()
    video_paths = glob.glob(f"{video_save_dir}*/*/*/*.mp4") 
    for i, video_path in enumerate(video_paths):
        vp = video_path.replace(video_save_dir, "")
        text_extract_save_path = vp.replace(".mp4", "_text_extract_monitors_ocrmac.json")
        text_extract_save_path = os.path.join(TEXT_EXTRACT_DIR, text_extract_save_path)
        if os.path.exists(text_extract_save_path):
            continue
        video_paths_to_run.append((video_path, text_extract_save_path))

    print(f"Video paths to run: {len(video_paths_to_run)}")

    with multiprocessing.Pool(cpu_count) as p:
        p.starmap(run_text_extract, video_paths_to_run)

if __name__ == "__main__":
    run_text_extract_multiprocessing(VIDEO_SAVE_DIR, cpu_count=8)
    video_path = os.path.join(VIDEO_SAVE_DIR, '2024/04/02/connor_mac_screen_recording-2024-04-02-20-11-57-UTC.mp4')
    # video_path = '/Users/connorparish/code/conbot/screencapture/screenshot_data/raw/2024/04/16/connor_mac_screen_recording-2024-04-16-19-34-45-UTC.mp4'
    # vp = video_path.replace(VIDEO_SAVE_DIR, "")
    # text_extract_save_path = vp.replace(".mp4", "_text_extract_monitors_ocrmac.json")
    # text_extract_save_path = os.path.join(TEXT_EXTRACT_DIR, text_extract_save_path)
    # run_text_extract(video_path=video_path, text_extract_save_path=text_extract_save_path)