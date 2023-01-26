import logging
import os

import pyautogui
from pynput.keyboard import Key
from pathlib import Path


class GestureRecognitionSettings:
    gesture_hot_key = Key.ctrl_l
    image_dir = Path('images')
    live_image_dir = Path('live_run_images')
    log_level = logging.DEBUG
    screen_res_x = pyautogui.size().width
    screen_res_y = pyautogui.size().height


class ModelConfig:
    data_dir = Path('image_dataset')
    batch_size = 16
    lr = 0.0001
    model_name = 'resnet18'
    classes = os.listdir(data_dir / 'train')
    latest_checkpoint_path = "lightning_logs/latest/best_checkpoint.ckpt"