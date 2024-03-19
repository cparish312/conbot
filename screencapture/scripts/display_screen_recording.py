import os
import time
from datetime import datetime
import glob
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import platform
from collections import deque
import threading
from threading import Thread

VIDEO_SAVE_DIR = "/Users/connorparish/code/conbot/screencapture/screenshot_data/raw/"

class VideoManager:
    def __init__(self, video_path, video_start_datetime, frame_num_start=None, frame_num_end=None):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.set_frame_nums(frame_num_start=frame_num_start, frame_num_end=frame_num_end)
        self.video_start_datetime = video_start_datetime

    def set_frame_nums(self, frame_num_start, frame_num_end):
        if frame_num_end is None:
            self.frame_num_start = frame_num_start
            self.frame_num_end = frame_num_start + self.total_frames
        else:
            self.frame_num_end = frame_num_end
            self.frame_num_start = frame_num_end - self.total_frames

    def get_frame(self, frame_num):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - self.frame_num_start)
        ret, frame = self.cap.read()
        if ret:
            return frame


class VideoTimelineManager:
    def __init__(self, video_save_dir=VIDEO_SAVE_DIR, min_frames_count=10000):
        self.create_video_paths_list(video_save_dir)
        self.min_frames_count = min_frames_count
        self.initalize_video_managers(self.min_frames_count)

    def create_video_paths_list(self, video_save_dir):
        self.video_paths_list = list()
        for video_path in glob.glob(f"{video_save_dir}*/*/*/*.mp4"):
            video_start_timestr_no_tz = video_path.split('-', 1)[1].rsplit('-', 1)[0]
            try:
                video_start_datetime = datetime.strptime(video_start_timestr_no_tz, '%Y-%m-%d-%H-%M-%S')
            except: 
                video_start_datetime = datetime.strptime(video_start_timestr_no_tz, '%Y-%m-%d-%H-%M')
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
        except ValueError:
            print(f"Error opening video file: {self.video_paths_list[add_path_index][1]}")
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


class ScrollableVideoViewer:
    def __init__(self, master, video_path, max_width=1280, max_height=960):
        self.master = master
        self.video_path = video_path
        self.max_width = max_width
        self.max_height = max_height
        self.scroll_frame_num = 0
        self.scroll_frame_num_var = tk.StringVar()
        self.scroll_frame_num_var.set(f"Frame: {self.scroll_frame_num}")
        self.display_frame_num = 0
        self.max_preload_buffer = 100
        
        self.video_timeline_manager = VideoTimelineManager()

        self.preloaded_frames = deque()
        self.preloaded_frames_lock = threading.Lock()
        # Preload first frame
        self.preload_frame(0, appendleft=False)
        self.setup_gui()

        self.exit_flag = False
        self.preload_thread = Thread(target=self.preload_frames)
        self.preload_thread.start()

    def preload_frame(self, frame_num, appendleft=False):
        frame = self.video_timeline_manager.get_frame(frame_num=frame_num)
        frame = self.resize_frame(frame, self.max_width, self.max_height)
        if appendleft:
            self.preloaded_frames.appendleft((frame_num, frame))
        else:
            self.preloaded_frames.append((frame_num, frame))
        return frame

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
            
            time.sleep(0.005)  # Sleep briefly to avoid overloading
        
    def setup_gui(self):
        self.master.title("Trackpad-Controlled Video Timeline")
        
        self.video_label = ttk.Label(self.master)
        self.video_label.pack()
        self.time_label = ttk.Label(self.master, textvariable=str(self.scroll_frame_num_var), font=("Arial", 24), anchor="e")
        self.time_label.pack()
        # Bind trackpad/mouse wheel event to the video navigation function
        self.bind_scroll_event()
        
        self.master.after(40, self.update_frame_periodically)

    def update_frame_periodically(self):
        # Check if the current scroll position has a preloaded frame
        with self.preloaded_frames_lock:
            for preloaded_frame_num, frame in self.preloaded_frames:
                if preloaded_frame_num == self.scroll_frame_num:
                    # Display the frame if it's preloaded
                    self.display_frame(frame)
                    break
            else:
                # If the frame isn't preloaded, preload and display it immediately
                # This will only happen if scrolling outside the preload range
                self.preloaded_frames.clear()
                frame = self.preload_frame(self.scroll_frame_num)
                self.display_frame(frame)

        # Schedule the next update
        if not self.exit_flag:
            self.master.after(10, self.update_frame_periodically)

    def display_frame(self, frame):
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def resize_frame(self, frame, max_width, max_height):
        height, width, _ = frame.shape
        if width > max_width or height > max_height:
            scaling_factor = min(max_width / width, max_height / height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return frame
    
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
    
        self.scroll_frame_num_var.set(f"Frame: {self.scroll_frame_num}")

    def on_scroll_up(self, event):
        # Linux scroll up
        if self.scroll_frame_num > 0:
            self.scroll_frame_num -= 1
        self.scroll_frame_num_var.set(f"Frame: {self.scroll_frame_num}")

    def on_scroll_down(self, event):
        # Linux scroll down
        self.scroll_frame_num += 1
        self.scroll_frame_num_var.set(f"Frame: {self.scroll_frame_num}")

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
