import time
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import platform
from collections import deque
from threading import Thread


class ScrollableVideoViewer:
    def __init__(self, master, video_path, max_width=640, max_height=480):
        self.master = master
        self.video_path = video_path
        self.max_width = max_width
        self.max_height = max_height
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_frame_time = time.time()
        
        self.frame_queue = deque(maxlen=10)  # Adjust size as needed
        self.exit_flag = False
        self.preload_thread = Thread(target=self.preload_frames)
        self.preload_thread.start()

        self.setup_gui()

        
    def preload_frames(self):
        """Preload frames into a deque in a background thread."""
        while not self.exit_flag:
            if len(self.frame_queue) < self.frame_queue.maxlen:
                next_frame_num = self.next_frame_to_preload()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_num)
                ret, frame = self.cap.read()
                if ret:
                    frame = self.resize_frame(frame, self.max_width, self.max_height)
                    self.frame_queue.append((next_frame_num, frame))
                else:
                    break
            else:
                time.sleep(0.01)  # Sleep briefly to avoid overloading
        
    def setup_gui(self):
        self.master.title("Trackpad-Controlled Video Timeline")
        
        self.video_label = ttk.Label(self.master)
        self.video_label.pack()
        
        # Bind trackpad/mouse wheel event to the video navigation function
        self.bind_scroll_event()
        
        self.scroll = tk.Scale(self.master, from_=0, to=self.total_frames, orient="horizontal", command=self.update_video_frame)
        self.scroll.pack(fill="x", expand=True)
        
        time.sleep(0.03)
        self.update_video_frame(0)
    
    def update_video_frame(self, frame_num):
        # Throttle the frame updates
        current_time = time.time()
        if current_time - self.last_frame_time < 0.03:  # Adjust as needed
            return
        self.last_frame_time = current_time

        # Check if the frame is preloaded
        frame = self.get_preloaded_frame(frame_num)
        if frame is None:
            # Frame wasn't preloaded, load it directly (consider background loading here)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
            ret, frame = self.cap.read()
            if not ret:
                return
            frame = self.resize_frame(frame, self.max_width, self.max_height)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def get_preloaded_frame(self, frame_num):
        for idx, (preloaded_frame_num, frame) in enumerate(self.frame_queue):
            if preloaded_frame_num == frame_num:
                # Found the preloaded frame
                return frame
        return None  # Frame not found
    
    def next_frame_to_preload(self):
        if self.frame_queue:
            return self.frame_queue[-1][0] + 1
        else:
            return int(self.scroll.get())
    
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
            self.scroll.set(self.scroll.get() - int(event.delta / 120))
        else:  # macOS
            self.scroll.set(self.scroll.get() + event.delta)
        self.update_video_frame(self.scroll.get())
    
    def on_scroll_up(self, event):
        # Linux scroll up
        self.scroll.set(self.scroll.get() - 1)
        self.update_video_frame(self.scroll.get())
    
    def on_scroll_down(self, event):
        # Linux scroll down
        self.scroll.set(self.scroll.get() + 1)
        self.update_video_frame(self.scroll.get())

    # When the window is closed, you should also gracefully exit the preload thread
    def on_window_close(self):
        self.exit_flag = True
        self.preload_thread.join()
        self.cap.release()
        self.master.destroy()

def main():
    root = tk.Tk()
    vid_path = '/Users/connorparish/code/conbot/screencapture/screenshot_data/raw/2024/03/18/connor_mac_screen_recording-2024-03-18-13-33-47-MDT.mp4'
    app = ScrollableVideoViewer(root, vid_path)
    root.mainloop()

if __name__ == "__main__":
    main()
