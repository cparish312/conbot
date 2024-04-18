import os
import json
import time
import tzlocal
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from datetime import time as dt_time
import glob
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import platform
from collections import deque
import threading
from threading import Thread

import numpy as np
import pandas as pd

from dataclasses import dataclass

import utils

VIDEO_SAVE_DIR = "/Users/connorparish/code/conbot/screencapture/screenshot_data/raw/"
VIDEO_CHANGES_DIR = "/Users/connorparish/code/conbot/screencapture/video_changes/"
TEXT_EXTRACT_DIR = "/Users/connorparish/code/conbot/screencapture/video_text_extract_ocrmac/"


FRAME_CHANGE_THRESHOLD = 0.05
local_timezone = tzlocal.get_localzone()

today_date = datetime.now().date() - timedelta(days=1)
time_at_2_10 = dt_time(hour=14, minute=10)  # Using 24-hour format for 2:10 PM
STARTING_TIME = datetime.combine(today_date, time_at_2_10).astimezone(local_timezone)

SHOW_ACTIVE_MONITOR = False

@dataclass
class VideoFrame:
    frame: np.array
    timestamp: datetime.timestamp
    text_df: pd.DataFrame

class VideoManager:
    def __init__(self, video_path, video_start_datetime, frame_num_start=None, frame_num_end=None, frame_change_threshold=FRAME_CHANGE_THRESHOLD):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_changes = self.load_frame_changes(video_path)
        self.unqiue_frame_to_frame_num = self.get_unique_frames(frame_change_threshold=frame_change_threshold)
        self.total_unique_frames = len(self.unqiue_frame_to_frame_num)
        print(f"{video_path} Total Frames: {self.total_frames} Unique Frames: {self.total_unique_frames}")
        self.set_frame_nums(frame_num_start=frame_num_start, frame_num_end=frame_num_end)
        self.video_start_datetime = video_start_datetime
        self.open_keystrokes(video_path)
        self.open_video_metadata(video_path)
        self.extracted_text_df = self.load_extracted_text(video_path)

    def set_frame_nums(self, frame_num_start, frame_num_end):
        if frame_num_end is None:
            self.frame_num_start = frame_num_start
            self.frame_num_end = frame_num_start + self.total_unique_frames - 1 # Needed for indexing
        else:
            self.frame_num_end = frame_num_end
            self.frame_num_start = frame_num_end - self.total_unique_frames

    def open_keystrokes(self, video_path):
        self.keystrokes_df = None
        keystrokes_path = video_path.replace("connor_mac_screen_recording", "connor_mac_keystrokes_recording").replace("mp4", "csv")
        if os.path.exists(keystrokes_path) and os.stat(keystrokes_path).st_size > 5:
            keystrokes_df = pd.read_csv(keystrokes_path)
            if 'x' not in keystrokes_df or 'y' not in keystrokes_df:
                self.keystrokes_df = None
                return
            keystrokes_df = keystrokes_df.dropna(subset=['x', 'y']) # only care about actions with location
            keystrokes_df = keystrokes_df.sort_values(by='_time')
            keystrokes_df['datetime'] = pd.to_datetime(keystrokes_df['_time']).dt.tz_localize('UTC', ambiguous='infer')
            self.keystrokes_df = keystrokes_df
        
    def open_video_metadata(self, video_path):
        self.metadata = None
        metadata_path = video_path.replace("connor_mac_screen_recording", "connor_mac_screen_recording_metadata").replace("mp4", "json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as infile:
                metadata_d = json.load(infile)
            self.metadata = metadata_d

    def get_last_keystroke_pos(self, timestamp):
        """Returns the last keystroke position before the given timestamp"""
        df = self.keystrokes_df.loc[self.keystrokes_df['datetime'] <= pd.to_datetime(timestamp)]
        if len(df) > 0:
            last_keystroke = df.iloc[-1]
            return last_keystroke['x'], last_keystroke['y']
        return None, None
    
    def load_frame_changes(self, video_path):
        vp = video_path.replace(VIDEO_SAVE_DIR, "")
        video_changes_file_name = vp.replace(".mp4", "_video_changes.json")
        video_changes_save_path = os.path.join(VIDEO_CHANGES_DIR, video_changes_file_name)
        if not os.path.exists(video_changes_save_path):
            return None
        
        with open(video_changes_save_path, 'r') as infile:
            return json.load(infile)
        
    def load_extracted_text(self, video_path):
        vp = video_path.replace(VIDEO_SAVE_DIR, "")
        text_extract_file_name = vp.replace(".mp4", "_text_extract_monitors_ocrmac.json")
        text_extract_save_path = os.path.join(TEXT_EXTRACT_DIR, text_extract_file_name)
        if not os.path.exists(text_extract_save_path):
            return None
        
        with open(text_extract_save_path, 'r') as infile:
            extracted_text = json.load(infile)
        
        def oc_dict_to_df(extracted_text):
            extracted_text_sections = list()
            for mon, mon_data in extracted_text.items():
                for frame_num, frame_data in mon_data.items():
                    for method, method_data in frame_data.items():
                        for oc, oc_data in method_data.items():
                            oc_tup = utils.str_to_tup(oc)
                            for ex in oc_data:
                                d = {'x' : ex[2][0] + oc_tup[0], 'y' : ex[2][1] + oc_tup[1],
                                    'r' : ex[2][2] + oc_tup[0], 'b' : ex[2][3] + oc_tup[1], 'conf' : ex[1], 
                                    'oc' : oc_tup, 'text' : ex[0], 'monitor' : int(mon), 'frame_num' : int(frame_num),
                                    'method' : method}
                                extracted_text_sections.append(d)
            return pd.DataFrame(extracted_text_sections)
        
        text_df = oc_dict_to_df(extracted_text)
        text_df['w'] = text_df['r'] - text_df['x']
        text_df['h'] = text_df['b'] - text_df['y']
        
        def get_index_to_frame_num_removed(text_df):
            index_to_frame_num_removed = {}
            active_index_to_frame_num = {}
            for frame_num in np.sort(text_df['frame_num'].unique()):
                frame_df = text_df.loc[text_df['frame_num'] == frame_num]
                frame_ocs = frame_df['oc'].unique()
                removed_area_indexes = set()
                for i, (start_frame_num, text_area) in active_index_to_frame_num.items():
                    within_oc = False
                    for oc in frame_ocs:
                        if utils.get_areas_overlap(oc, text_area) is not None:
                            within_oc = True
                    if within_oc:
                        index_to_frame_num_removed[i] = frame_num
                        removed_area_indexes.add(i)

                for i in removed_area_indexes:
                    del active_index_to_frame_num[i]
                
                for i, row in frame_df.iterrows():
                    active_index_to_frame_num[i] = (frame_num, (row['x'], row['y'], row['w'], row['h']))

            return index_to_frame_num_removed

        index_to_frame_num_removed = get_index_to_frame_num_removed(text_df=text_df)
        text_df['frame_num_removed'] = text_df.index.map(index_to_frame_num_removed).fillna(value=max(text_df['frame_num'].unique()) + 1)
        return text_df
    
    def get_unique_frames(self, frame_change_threshold):
        if self.frame_changes is None:
            return {i : i for i in range(self.total_frames)}
        unqiue_frame_to_frame_num = {0 : 0}
        unique_frame_c = 1
        cum_diff_percent = 0
        for i, fc in enumerate(self.frame_changes):
            cum_diff_percent += fc['diff_percent']
            if cum_diff_percent >= frame_change_threshold:
                unqiue_frame_to_frame_num[unique_frame_c] = i + 1
                unique_frame_c += 1
                cum_diff_percent = 0
        return unqiue_frame_to_frame_num

    def get_active_monitor(self, frame, frame_timestamp):
        if self.keystrokes_df is None or self.metadata is None:
            return None
        
        x, y = self.get_last_keystroke_pos(frame_timestamp)
        if x is None or y is None:
            return frame
        for monitor in self.metadata['monitors']:
            if x >= monitor['left'] and x <= monitor['left'] + monitor['width'] \
                and y >= monitor['top'] and y <= monitor['top'] + monitor['height']:
                    return (monitor['left'], monitor['top'], monitor['width'], monitor['height'])
                    # return frame[monitor['top'] : monitor['top'] + monitor['height'], monitor['left'] : monitor['left'] + monitor['width']]
            
        raise ValueError(f"Position ({x}, {y}) outside of monitors.")

    def get_frame(self, unique_frame_num):
        frame_num = self.unqiue_frame_to_frame_num[self.frame_num_end - unique_frame_num]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            frame_timestamp = self.video_start_datetime + timedelta(milliseconds=self.cap.get(cv2.CAP_PROP_POS_MSEC))
            frame_text_df = None
            if self.extracted_text_df is not None:
                frame_text_df = self.extracted_text_df.loc[(self.extracted_text_df['frame_num'] <= frame_num) & (self.extracted_text_df['frame_num_removed'] > frame_num)]
            if SHOW_ACTIVE_MONITOR:
                active_monitor = self.get_active_monitor(frame, frame_timestamp)
                if active_monitor is not None:
                    frame = frame[active_monitor[1]:active_monitor[1] + active_monitor[3], active_monitor[0]:active_monitor[0] + active_monitor[2]]
                    if frame_text_df is not None:
                        frame_text_df = frame_text_df.loc[(frame_text_df['x'] >= active_monitor[0]) & (frame_text_df['x'] <= (active_monitor[0] + active_monitor[2])) \
                                                        & (frame_text_df['y'] >= active_monitor[1]) & (frame_text_df['y'] <= (active_monitor[1] + active_monitor[3]))]
                        frame_text_df['x'] -= active_monitor[0]
                        frame_text_df['y'] -= active_monitor[1]
            return VideoFrame(frame=frame, timestamp=frame_timestamp, text_df=frame_text_df)

class VideoTimelineManager:
    def __init__(self, video_save_dir=VIDEO_SAVE_DIR, min_frames_count=100):
        self.create_video_paths_list(video_save_dir)
        self.min_frames_count = min_frames_count
        self.initalize_video_managers(self.min_frames_count)

    def convert_timezone_to_local(self, video_start_datetime, video_timezone):
        if video_timezone == "MDT": # All videos should now be stored in UTC
            video_timezone = "America/Denver"
        video_timezone = ZoneInfo(video_timezone)
        if video_timezone == local_timezone:
            return video_start_datetime.replace(tzinfo=video_timezone)
        else:
            return video_start_datetime.replace(tzinfo=video_timezone).astimezone(local_timezone)

    def create_video_paths_list(self, video_save_dir):
        self.video_paths_list = list()
        for video_path in glob.glob(f"{video_save_dir}*/*/*/*.mp4"):
            video_start_timestr_no_tz = video_path.split('-', 1)[1].rsplit('-', 1)[0]
            video_start_tz = video_path.split('-', 1)[1].rsplit('-', 1)[1].split('.')[0]
            try:
                video_start_datetime = datetime.strptime(video_start_timestr_no_tz, '%Y-%m-%d-%H-%M-%S')
            except: 
                video_start_datetime = datetime.strptime(video_start_timestr_no_tz, '%Y-%m-%d-%H-%M')
            video_start_datetime = self.convert_timezone_to_local(video_start_datetime=video_start_datetime, video_timezone=video_start_tz)
            if STARTING_TIME is not None and video_start_datetime > STARTING_TIME:
                continue
            self.video_paths_list.append((video_start_datetime, video_path))

        print("Total Video Paths", len(self.video_paths_list))
        self.video_paths_list.sort(key=lambda item: item[0], reverse=True)

    def add_video_manager(self, newer=False):
        frame_num_start = None
        frame_num_end = None
        if len(self.video_managers) == 0:
            add_path_index = 0
            frame_num_start = 0
        elif newer:
            add_path_index = self.video_managers[0][0] - 1
            frame_num_end = self.video_managers[0][1].frame_num_start
        else:
            add_path_index = self.video_managers[-1][0] + 1
            frame_num_start = self.video_managers[-1][1].frame_num_end
            
        try:
            new_video_manager = VideoManager(self.video_paths_list[add_path_index][1], 
                                            self.video_paths_list[add_path_index][0],
                                            frame_num_start=frame_num_start,
                                            frame_num_end=frame_num_end)
        except ValueError as e:
            print(e)
            # print(f"Error opening video file: {self.video_paths_list[add_path_index][1]}")
            del self.video_paths_list[add_path_index]
            self.add_video_manager(newer=newer)
            return

        if newer:
            self.video_managers.appendleft((add_path_index, new_video_manager))
        else:
            self.video_managers.append((add_path_index, new_video_manager))

    def initalize_video_managers(self, min_frames_count):
        self.video_managers = deque()
        self.add_video_manager()
        while self.video_managers[-1][1].frame_num_end < min_frames_count:
            self.add_video_manager()
        print("Number of video managers: ", len(self.video_managers))

    def get_frame(self, frame_num):
        for _, video_manager in self.video_managers:
            if frame_num >= video_manager.frame_num_start and frame_num < video_manager.frame_num_end:
                return video_manager.get_frame(frame_num)
        
        self.add_video_manager()
        return self.get_frame(frame_num)
                

class ScrollableVideoViewer:
    def __init__(self, master, video_path, max_width=1536, max_height=1152):
        self.master = master
        self.video_path = video_path
        self.max_width = max_width
        self.max_height = max_height
        self.scroll_frame_num = 0
        self.scroll_frame_num_var = tk.StringVar()
        self.max_preload_buffer = 100
        
        self.video_timeline_manager = VideoTimelineManager()

        self.preloaded_frames = deque()
        self.preloaded_frames_lock = threading.Lock()

        # For mouse dragging
        self.dragging = False
        self.drag_start = None
        self.drag_end = None

        # Preload first frame
        self.preload_frame(0, appendleft=False)
        self.setup_gui()

        self.exit_flag = False
        self.preload_thread = Thread(target=self.preload_frames)
        self.preload_thread.start()

    def preload_frame(self, frame_num, appendleft=False):
        videoframe = self.video_timeline_manager.get_frame(frame_num=frame_num)
        self.resize_frame(videoframe, self.max_width, self.max_height)
        self.add_text_boxes(videoframe)
        if appendleft:
            self.preloaded_frames.appendleft((frame_num, videoframe))
        else:
            self.preloaded_frames.append((frame_num, videoframe))
        return videoframe
    
    def add_text_boxes(self, videoframe):
        if videoframe.text_df is not None:
            for i, row in videoframe.text_df.iterrows():
                cv2.rectangle(videoframe.frame, (int(row['x']), int(row['y'])), (int(row['r']), int(row['b'])), (0, 255, 0), 1)
    
    def resize_frame(self, videoframe, max_width, max_height):
        height, width, _ = videoframe.frame.shape
        if width > max_width or height > max_height:
            scaling_factor = min(max_width / width, max_height / height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            videoframe.frame = cv2.resize(videoframe.frame, new_size, interpolation=cv2.INTER_AREA)
            if videoframe.text_df is not None:
                videoframe.text_df.loc[:, 'x'] = videoframe.text_df['x'] * scaling_factor
                videoframe.text_df.loc[:, 'y'] = videoframe.text_df['y'] * scaling_factor
                videoframe.text_df.loc[:, 'w'] = videoframe.text_df['w'] * scaling_factor
                videoframe.text_df.loc[:, 'h'] = videoframe.text_df['h'] * scaling_factor
                videoframe.text_df.loc[:, 'r'] = videoframe.text_df['r'] * scaling_factor
                videoframe.text_df.loc[:, 'b'] = videoframe.text_df['b'] * scaling_factor

    def preload_frames(self):
        """Preload frames into a deque in a background thread."""
        while not self.exit_flag:
            with self.preloaded_frames_lock:
                # If need to extend preload window to lower number frames
                if self.preloaded_frames[0][0] != 0 and self.scroll_frame_num - self.preloaded_frames[0][0] < self.max_preload_buffer:
                    next_left_frame_num = self.preloaded_frames[0][0] - 1
                    self.preload_frame(next_left_frame_num, appendleft=True)
                    # print('L', next_left_frame_num)
                # If need to extend preload window to higher number frames
                if self.preloaded_frames[-1][0] - self.scroll_frame_num < self.max_preload_buffer:
                    next_right_frame_num = self.preloaded_frames[-1][0] + 1
                    self.preload_frame(next_right_frame_num, appendleft=False)
                    # print('R', next_right_frame_num)
            
            time.sleep(0.0001)  # Sleep briefly to avoid overloading
        
    def setup_gui(self):
        self.master.title("Trackpad-Controlled Video Timeline")
        
        self.video_label = ttk.Label(self.master)
        self.video_label.pack()
        self.video_label.bind("<Button-1>", self.start_drag)
        self.video_label.bind("<B1-Motion>", self.on_drag)
        self.video_label.bind("<ButtonRelease-1>", self.end_drag)

        self.time_label = ttk.Label(self.master, textvariable=str(self.scroll_frame_num_var), font=("Arial", 24), anchor="e")
        self.time_label.pack()
        self.bind_scroll_event()
        
        self.displayed_frame_num = -1
        self.master.after(40, self.update_frame_periodically)

    def update_frame_periodically(self):
        # Check if the current scroll position has a preloaded frame
        if self.scroll_frame_num != self.displayed_frame_num:
            with self.preloaded_frames_lock:
                for preloaded_frame_num, videoframe in self.preloaded_frames:
                    if preloaded_frame_num == self.scroll_frame_num:
                        # Display the frame if it's preloaded
                        self.display_frame(videoframe, preloaded_frame_num)
                        break
                else:
                    # If the frame isn't preloaded, preload and display it immediately
                    # This will only happen if scrolling outside the preload range
                    self.preloaded_frames.clear()
                    videoframe = self.preload_frame(self.scroll_frame_num)
                    self.display_frame(videoframe, preloaded_frame_num)

        # Schedule the next update
        if not self.exit_flag:
            self.master.after(10, self.update_frame_periodically)

    def display_frame(self, videoframe, frame_num):
        print(frame_num)
        print('num preloaded', len(self.preloaded_frames))
        cv2image = cv2.cvtColor(videoframe.frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.scroll_frame_num_var.set(f"Time: {videoframe.timestamp}")
        self.displayed_videoframe = videoframe
        self.displayed_frame_num = frame_num
    
    def bind_scroll_event(self):
        # Detect platform and bind the appropriate event
        os_name = platform.system()
        if os_name == "Linux":
            self.master.bind("<Button-4>", self.on_scroll_up)
            self.master.bind("<Button-5>", self.on_scroll_down)
        else:  # Windows and macOS
            self.master.bind("<MouseWheel>", self.on_mouse_wheel)
    
    def on_mouse_wheel(self, event):
        # Windows and macOS handle scroll direction differently
        if platform.system() == "Windows":
            self.scroll_frame_num += int(event.delta / 120)
        else:  # macOS
            self.scroll_frame_num -= int(event.delta)

        if self.scroll_frame_num < 0:
            self.scroll_frame_num = 0

    def on_scroll_up(self, event):
        # Linux scroll up
        if self.scroll_frame_num > 0:
            self.scroll_frame_num -= 1

    def on_scroll_down(self, event):
        # Linux scroll down
        self.scroll_frame_num += 1

    def start_drag(self, event):
        self.dragging = True
        self.drag_start = (event.x, event.y)

    def on_drag(self, event):
        if self.dragging:
            self.drag_end = (event.x, event.y)

    def end_drag(self, event):
        if not self.dragging:
            return
        self.dragging = False
        self.copy_texts_within_drag_area()

    def rectangles_overlap(self, rect1, rect2):
        """Check if two rectangles overlap. Rectangles are defined as (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = rect1
        rx1, ry1, rx2, ry2 = rect2
        return not (rx1 > x2 or rx2 < x1 or ry1 > y2 or ry2 < y1)

    def copy_texts_within_drag_area(self):
        x1 = min(self.drag_start[0], self.drag_end[0]) 
        y1 = min(self.drag_start[1], self.drag_end[1])
        x2 = max(self.drag_start[0], self.drag_end[0]) 
        y2 = max(self.drag_start[1], self.drag_end[1])

        text_df = self.displayed_videoframe.text_df
        selected_texts = []
        if text_df is not None:
            texts_in_area = text_df.apply(lambda row: self.rectangles_overlap((x1, y1, x2, y2), (row['x'], row['y'], row['x'] + row['w'], row['y'] + row['h'])), axis=1)
            overlapping_texts = text_df[texts_in_area]
            selected_texts.extend(overlapping_texts['text'].tolist())
        
        if selected_texts:
            self.copy_to_clipboard("\n".join(selected_texts))
        else:
            messagebox.showinfo("No Text Selected", "No text found within the selected area.")

    def copy_to_clipboard(self, text):
        """Copy provided text to clipboard."""
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        messagebox.showinfo("Text Copied", f"Text has been copied to clipboard:\n{text}")


    # When the window is closed, you should also gracefully exit the preload thread
    def on_window_close(self):
        self.exit_flag = True
        self.preload_thread.join()
        for _, video_manager in self.video_timeline_manager.video_managers:
            video_manager.cap.release()
        self.master.destroy()

def main():
    root = tk.Tk()
    vid_path = '/Users/connorparish/code/conbot/screencapture/screenshot_data/raw/2024/03/18/connor_mac_screen_recording-2024-03-18-13-33-47-MDT.mp4'
    app = ScrollableVideoViewer(root, vid_path)
    root.protocol("WM_DELETE_WINDOW", app.on_window_close)  # Handle window close event
    root.mainloop()

if __name__ == "__main__":
    main()
