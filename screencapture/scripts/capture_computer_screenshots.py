"""Script to record screen on computer"""

import os
import time
import datetime

import cv2
import mss
import numpy as np
from pynput import keyboard, mouse

import utils

SAVE_DIR = "../screenshot_data/raw/"
SHOTS_PER_SECOND=1
RESOLUTION_FACTOR=1.2 # For decreasing size of videos

recording = True

def stop_recording():
    global recording
    recording = False

key_func_map = {
    '<ctrl>+q': stop_recording,
}

def start_video_capture(save_dir):
    global recording 
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # Capture all screens as one video
        now = datetime.datetime.now()
        now_dir = os.path.join(save_dir, f'{now.year}/{now.month}/{now.day}')
        utils.make_dirs(now_dir)
        now_timestr = now.strftime("%Y%m%d-%H%M%S")
        video_filename = os.path.join(now_dir, f'connor_mac_screen_recording-{now_timestr}.mp4')

        # Define the codec and create VideoWriter object
        # Get width and height of frame (combined screen doesn't match desribed width and height)
        frame = np.array(sct.grab(monitor))
        height, width = frame.shape[0], frame.shape[1]
        adjusted_height, adjusted_width = (int(height//RESOLUTION_FACTOR), int(width//RESOLUTION_FACTOR))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Efficient codec for .mp4
        print(adjusted_height, adjusted_width)
        out = cv2.VideoWriter(video_filename, fourcc, SHOTS_PER_SECOND, (adjusted_width, adjusted_height))

        # Calculate adjusting ratios for mouse x, y (resolution reported by pynput is for monitor[0] reported width height)
        mouse_x_ratio = adjusted_width / monitor["width"]
        mouse_y_ratio = adjusted_height / monitor["height"]

        print(f"Starting video capture: {video_filename}")
        listener = keyboard.GlobalHotKeys(key_func_map)
        listener.start()
        mouse_controller = mouse.Controller()
    
        while recording:
            start_time = datetime.datetime.now()
            
            screenshot = sct.grab(monitor)
            mouse_x, mouse_y = mouse_controller.position
            mouse_x, mouse_y = mouse_x - monitor['left'], mouse_y - monitor['top']
            mouse_x, mouse_y = (int(mouse_x*mouse_x_ratio), int(mouse_y*mouse_y_ratio))

            frame = np.array(screenshot)
            # Convert from BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, (adjusted_width, adjusted_height))

            # To draw mouse in screenshot
            frame = cv2.circle(frame, (int(mouse_x), int(mouse_y)), 100, (0, 255, 0), -1)
            out.write(frame)

            # Frame rate control
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            sleep_time = max(1.0 / SHOTS_PER_SECOND - elapsed, 0)
            cv2.imshow("combined screens", frame)
            cv2.waitKey(int(sleep_time * 1000))

        out.release()
        cv2.destroyAllWindows()  # Ensure all windows are closed
        listener.stop()
        print(f"Video saved: {video_filename}")


if __name__ == '__main__':
    start_video_capture(SAVE_DIR)
    
        