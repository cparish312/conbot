"""Script that tests out multiple ways of doing text extract on video"""

import os
import glob
import json
import multiprocessing
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import pytesseract

import utils

VIDEO_SAVE_DIR = "/Users/connorparish/code/conbot/screencapture/screenshot_data/raw/"
VIDEO_CHANGES_DIR = "/Users/connorparish/code/conbot/screencapture/video_changes/"
TEXT_EXTRACT_DIR = "/Users/connorparish/code/conbot/screencapture/video_text_extract/"

class VideoTextExtractor:
    def __init__(self, video_path, text_extract_save_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {self.total_frames}")
        self.frame_changes = self.load_frame_changes(video_path)
        if self.frame_changes is None:
            raise ValueError(f"No frame changes metadata for: {video_path}")
    
        self.extract_text(text_extract_save_path)
            
    def load_frame_changes(self, video_path):
        vp = video_path.replace(VIDEO_SAVE_DIR, "")
        video_changes_file_name = vp.replace(".mp4", "_video_changes.json")
        video_changes_save_path = os.path.join(VIDEO_CHANGES_DIR, video_changes_file_name)
        if not os.path.exists(video_changes_save_path):
            return None
        
        with open(video_changes_save_path, 'r') as infile:
            return json.load(infile)

    def extract_text_from_frame(self, frame):
        return pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
    
    def extract_text_from_frame_threshhold(self, frame):
        frame = cv2.medianBlur(frame, 3)
        # frame =  cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        frame_thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2) 
        return pytesseract.image_to_data(frame_thresh, output_type=pytesseract.Output.DICT)
    
    def object_based_frame_extraction_color_modes(self, frame_gray, count_thresh=40000, contour_area_thresh=100000):
        colors, counts = np.unique(frame_gray, return_counts=True)
        query_colors = {color for color, count in zip(colors, counts) if count >= count_thresh}
        objects_mask = np.zeros_like(frame_gray)
        object_contours = list()
        for qc in query_colors:
            mask = cv2.inRange(frame_gray, np.array([qc]), np.array([qc]))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # If the contour is too small, ignore it
                if cv2.contourArea(contour) < contour_area_thresh:
                    continue
                # Compute the bounding box for the contour, draw it on the frame
                (x, y, w, h) = cv2.boundingRect(contour)
                object_contours.append((x, y, w, h))
                objects_mask[y:y + h, x: x+w] = 1
        return object_contours, objects_mask
    
    def object_based_frame_extraction_edge_detection(self, frame_gray, contour_area_thresh=100000):
        objects_mask = np.zeros_like(frame_gray)
        edged = cv2.Canny(frame_gray, 50, 150, apertureSize=5, L2gradient=True) 
        dilated = cv2.dilate(edged, None, iterations=10)   
        max_contour_size = frame_gray.shape[0] * frame_gray.shape[1] / 2.2
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        object_contours = list()
        for contour in contours:
            # If the contour is too small, ignore it
            if cv2.contourArea(contour) < contour_area_thresh or cv2.contourArea(contour) > max_contour_size:
                continue
            # Compute the bounding box for the contour, draw it on the frame
            (x, y, w, h) = cv2.boundingRect(contour)
            object_contours.append((x, y, w, h))
            objects_mask[y:y + h, x: x+w] = 1
        return object_contours, objects_mask
    
    def contours_to_text(self, contours, mask, frame):
        oc_to_text = dict()
        for oc in contours:
            (x, y, w, h) = oc
            contour_frame = frame[y:y + h, x: x+w]
            contour_text = self.extract_text_from_frame(contour_frame)
            oc_to_text[self.tuple_to_str((oc))] = contour_text

        # Run on rest of image that was not part of object contour
        frame_copy = frame.copy()
        frame_copy[mask>0]=255
        contour_text = self.extract_text_from_frame(frame_copy)
        oc_to_text[self.tuple_to_str((0, 0, frame.shape[1], frame.shape[0]))] = contour_text
        return oc_to_text
    
    def tuple_to_str(self, tup):
        s = "("
        for v in tup:
            s += str(v) + ","
        return s[:-1] + ")"

    def extract_text(self, text_extract_save_path):
        self.frame_num_to_extracted_text = defaultdict(lambda: defaultdict(dict))
        frame_num = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_num_to_extracted_text[frame_num]['brute'][self.tuple_to_str((0, 0, frame.shape[1], frame.shape[0]))] = self.extract_text_from_frame(frame)
        self.frame_num_to_extracted_text[frame_num]['brute_gray'][self.tuple_to_str((0, 0, frame.shape[1], frame.shape[0]))] = self.extract_text_from_frame(frame_gray)
        self.frame_num_to_extracted_text[frame_num]['brute_threshold'][self.tuple_to_str((0, 0, frame.shape[1], frame.shape[0]))] = self.extract_text_from_frame_threshhold(frame_gray)

        object_contours, objects_mask = self.object_based_frame_extraction_edge_detection(frame_gray)
        self.frame_num_to_extracted_text[frame_num]['obj_edge_detect'] = self.contours_to_text(object_contours, objects_mask, frame)

        object_contours, objects_mask = self.object_based_frame_extraction_color_modes(frame_gray)
        self.frame_num_to_extracted_text[frame_num]['obj_color_mode'] = self.contours_to_text(object_contours, objects_mask, frame)

        while True:
            # Read the next frame
            ret, frame = self.cap.read()
            frame_num += 1
            if not ret:
                break  # Reached the end of the video

            if self.frame_changes[frame_num - 1]['diff_percent'] < 0.005:
                continue
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_num_to_extracted_text[frame_num]['brute'][self.tuple_to_str((0, 0, frame.shape[1], frame.shape[0]))] = self.extract_text_from_frame(frame)
            self.frame_num_to_extracted_text[frame_num]['brute_gray'][self.tuple_to_str((0, 0, frame.shape[1], frame.shape[0]))] = self.extract_text_from_frame(frame_gray)
            self.frame_num_to_extracted_text[frame_num]['brute_threshold'][self.tuple_to_str((0, 0, frame.shape[1], frame.shape[0]))] = self.extract_text_from_frame_threshhold(frame_gray)

            # Run brute witin the compression change boxes
            cb = self.frame_changes[frame_num - 1]['change_box']
            change_box = frame[cb[1]: cb[3], cb[0]: cb[2]]
            change_box_gray = frame_gray[cb[1]: cb[3], cb[0]: cb[2]]
            self.frame_num_to_extracted_text[frame_num]['brute_change_box'][self.tuple_to_str((cb[0], cb[1], cb[2], cb[3]))] = self.extract_text_from_frame(change_box)
            self.frame_num_to_extracted_text[frame_num]['brute_gray_change_box'][self.tuple_to_str((cb[0], cb[1], cb[2], cb[3]))] = self.extract_text_from_frame(change_box_gray)

            # Run object based extraction on change boxes
            object_contours, objects_mask = self.object_based_frame_extraction_edge_detection(change_box_gray)
            self.frame_num_to_extracted_text[frame_num]['obj_edge_detect'] = self.contours_to_text(object_contours, objects_mask, change_box)

            object_contours, objects_mask = self.object_based_frame_extraction_color_modes(change_box_gray)
            self.frame_num_to_extracted_text[frame_num]['obj_color_mode'] = self.contours_to_text(object_contours, objects_mask, change_box)

        utils.make_dirs("/".join(text_extract_save_path.split("/")[:-1]))
        with open(text_extract_save_path, 'w') as outfile:
            json.dump(self.frame_num_to_extracted_text, outfile)

def run_text_extract(video_path, text_extract_save_path):
    VideoTextExtractor(video_path=video_path, text_extract_save_path=text_extract_save_path)

def run_text_extract_multiprocessing(video_save_dir, cpu_count=os.cpu_count()):
    video_paths_to_run = list()
    video_paths = glob.glob(f"{video_save_dir}*/*/*/*.mp4") 
    for i, video_path in enumerate(video_paths):
        vp = video_path.replace(video_save_dir, "")
        text_extract_save_path = vp.replace(".mp4", "_text_extract.json")
        text_extract_save_path = os.path.join(TEXT_EXTRACT_DIR, text_extract_save_path)
        if os.path.exists(text_extract_save_path):
            continue
        video_paths_to_run.append((video_path, text_extract_save_path))

    print(f"Video paths to run: {len(video_paths_to_run)}")

    with multiprocessing.Pool(cpu_count) as p:
        p.starmap(run_text_extract, video_paths_to_run)

if __name__ == "__main__":
    run_text_extract_multiprocessing(VIDEO_SAVE_DIR, cpu_count=8)