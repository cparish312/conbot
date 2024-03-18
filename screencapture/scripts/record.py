"""Script created by Connor Parish for recording actions on a computer."""
import os
import time
import threading

import cv2
import mss
import numpy as np
import pandas as pd
from pynput import keyboard, mouse

import utils

SAVE_DIR = "/Users/connorparish/code/conbot/screencapture/screenshot_data/raw/"
SHOTS_PER_SECOND=1
RESOLUTION_FACTOR=1.2 # For decreasing size of videos
RECORDING_STOP_DELAY=60 # In seconds

user_active = True
screen_recording = False

actions_list_lock = threading.Lock()

actions = list()
last_action_time = time.time()

def on_press(key):
    global user_active
    global actions
    global last_action_time

    try:
        press_object = {
            'action':'kp', 
            'button':key.char, 
            '_time': time.time()
        }
    except AttributeError:
        press_object = {
            'action':'kp', 
            'button':str(key), 
            '_time': time.time()
        }
    with actions_list_lock:
        actions.append(press_object)
    user_active = True
    last_action_time = press_object['_time']

def on_release(key):
    global user_active
    global actions
    global last_action_time

    try:
        release_object = {
            'action':'kr', 
            'button':key.char, 
            '_time': time.time()
        }
    except AttributeError:
        release_object = {
            'action':'kr', 
            'button':str(key), 
            '_time': time.time()
        }
    with actions_list_lock:
        actions.append(release_object)
    user_active = True
    last_action_time = release_object['_time']

def on_click(x, y, button, pressed):
    global user_active
    global actions
    global last_action_time
    click_object = {
        'action':'c' if pressed else 'uc', 
        'button':str(button), 
        'x':x, 
        'y':y, 
        '_time':time.time()
    }
    
    with actions_list_lock:
        actions.append(click_object)
    user_active = True
    last_action_time = click_object['_time']

def on_scroll(x, y, dx, dy):
    global user_active
    global actions
    global last_action_time
    scroll_object = {
        'action': 'scroll', 
        'dy': int(dy), 
        'dx': int(dx), 
        'x':x, 
        'y':y, 
        '_time': time.time()
    }

    def add_to_scroll(scroll_object: dict) -> bool:
        if len(actions) == 0:
            return False
        last_action = actions[-1]
        return (last_action["action"] == "scroll" and 
            last_action['x'] == scroll_object['x'] and 
            last_action["y"] == scroll_object["y"] and
            (last_action["dy"] > 0) == (scroll_object["dy"] > 0) and 
            (last_action["dx"] > 0) == (scroll_object["dx"] > 0))
    
    if scroll_object["dy"] != 0 or scroll_object["dx"] != 0:
        with actions_list_lock:
            if add_to_scroll(scroll_object=scroll_object):
                actions[-1]["dy"] += scroll_object["dy"]
                actions[-1]["dx"] += scroll_object["dx"]
            else:
                actions.append(scroll_object)
    user_active = True
    last_action_time = scroll_object['_time']

def on_move(x, y):
    global user_active
    global last_action_time
    user_active = True
    last_action_time = time.time() 

def save_actions(save_file: str, monitor, adjusted_height, adjusted_width):
    global actions

    with actions_list_lock:
        actions_df = pd.DataFrame(actions)
        actions.clear()

    if {"x", "y"} <= set(actions_df.columns):
        # Calculate adjusting ratios for mouse x, y (resolution reported by pynput is for monitor[0] reported width height)
        mouse_x_ratio = adjusted_width / monitor["width"]
        mouse_y_ratio = adjusted_height / monitor["height"]
        actions_df['x'] = actions_df['x'] - monitor['left']
        actions_df['y'] = actions_df['y'] - monitor['top']
        actions_df['x'] = actions_df['x']*mouse_x_ratio
        actions_df['y'] = actions_df['y']*mouse_y_ratio
        actions_df['x'] = actions_df['x'].round(2)
        actions_df['y'] = actions_df['y'].round(2)

    actions_df.to_csv(save_file, index=False)

def get_monitor_count():
    """Returns the current number of monitors."""
    with mss.mss() as sct:
        return len(sct.monitors)

def start_video_capture(save_dir: str):
    global screen_recording 
    global last_action_time
    global user_active

    with mss.mss() as sct:
        monitor = sct.monitors[0]  # Capture all screens as one video
        # Get number of monitors to ensure no change during recording
        num_monitors = len(sct.monitors)

        # Get start time
        video_start_time = time.time()
        video_start_timestr = time.strftime('%Y-%m-%d-%H-%M-%S-%Z', time.localtime(video_start_time))
        video_start_timestr_split = video_start_timestr.split('-')
        today_save_dir = os.path.join(save_dir, f'{video_start_timestr_split[0]}/{video_start_timestr_split[1]}/{video_start_timestr_split[2]}')
        utils.make_dirs(today_save_dir)
        video_filename = os.path.join(today_save_dir, f'connor_mac_screen_recording-{video_start_timestr}.mp4')
        keystrokes_filename = os.path.join(today_save_dir, f'connor_mac_keystrokes_recording-{video_start_timestr}.csv')

        # Define the codec and create VideoWriter object
        # Get width and height of frame (combined screen doesn't match desribed width and height)
        frame = np.array(sct.grab(monitor))
        height, width = frame.shape[0], frame.shape[1]
        adjusted_height, adjusted_width = (int(height//RESOLUTION_FACTOR), int(width//RESOLUTION_FACTOR))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Efficient codec for .mp4
        out = cv2.VideoWriter(video_filename, fourcc, SHOTS_PER_SECOND, (adjusted_width, adjusted_height))

        print(f"Starting video capture: {video_filename}")
        # mouse_controller = mouse.Controller()

        screen_recording = True
        while screen_recording:
            if get_monitor_count() != num_monitors:
                print("Number of monitors have changed. Creating new recording.")
                out.release()
                cv2.destroyAllWindows()  # Ensure all windows are closed
                save_actions(keystrokes_filename, monitor, adjusted_height, adjusted_width)
                print(f"Video and keystrokes saved: {video_filename}")
                start_video_capture(save_dir=save_dir)
                return
            
            start_time = time.time()
            
            screenshot = sct.grab(monitor)
            # mouse_x, mouse_y = mouse_controller.position

            frame = np.array(screenshot)
            # Convert from BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, (adjusted_width, adjusted_height))

            out.write(frame)

            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = max(1.0 / SHOTS_PER_SECOND - elapsed, 0)
            cv2.waitKey(int(sleep_time * 1000))

            if time.time() - last_action_time > RECORDING_STOP_DELAY:
                screen_recording = False
                user_active = False

        out.release()
        cv2.destroyAllWindows()  # Ensure all windows are closed
        save_actions(keystrokes_filename, monitor, adjusted_height, adjusted_width)

        print(f"Video and keystrokes saved: {video_filename}")


def record(save_dir: str):
    global user_active
    global screen_recording
    keyboard_listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)

    mouse_listener = mouse.Listener(
            on_click=on_click,
            on_scroll=on_scroll,
            on_move=on_move)

    keyboard_listener.start()
    time.sleep(1) # Needed for bug in pynput (lazy loading within pyobjc)
    mouse_listener.start()
    
    while True:
        if user_active and not screen_recording:
            start_video_capture(save_dir=save_dir)

    # keyboard_listener.join()
    # mouse_listener.join()

if __name__ == "__main__":
    record(SAVE_DIR)


