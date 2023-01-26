import os
import sys
from pathlib import Path

import torch
from PIL import Image
import argparse

from cfg import ModelConfig
from command_executor import CommandExecutor
from gesture_mapping import GestureMapping
from models.shape_detector_model import ShapeDetectionModel

from queue import Queue

from cfg import GestureRecognitionSettings
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener

from models.train import start_training
from split_dir_into_train_test import copy_to_train_test_subdirectories, clean_old_train_test_directories
from utils import save_drawing, get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = get_logger(__name__)


# add current directory to sys path if not present
THIS_DIRECTORY = str(Path(__file__).parent)
if THIS_DIRECTORY not in sys.path:
    sys.path.append(THIS_DIRECTORY)


class InteractionState:
    gesture_capture = False
    last_mouse_sequence = []
    current_mouse_sequence = []


state = InteractionState()
queue = Queue()
queue.put(state)


class Mode:
    CREATE_DATASET = "create_dataset"
    LIVE = "live"


class GestureRecognitionEngine:

    def __init__(self,
                 mode: str
                 ):
        self.mode = mode
        if mode == Mode.LIVE:
            log.info("Live Mode")
            log.info("Initialising Shape Detector Model")
            self.model = ShapeDetectionModel.load_from_checkpoint(ModelConfig.latest_checkpoint_path,
                                                                  model_cfg=ModelConfig)
            self.model = self.model.to(device)
            log.info("Initialing Gesture Command Mappings")
            self.command_executor = CommandExecutor.from_gesture_mapping(GestureMapping)

            GestureRecognitionSettings.live_image_dir.mkdir(exist_ok=True)
        elif mode == Mode.CREATE_DATASET:
            log.info("Dataset Creation Mode")
            self.dataset_gesture_name = input("Enter the name of the gesture (eg: circle, zigzag etc) : ")
            filepath = GestureRecognitionSettings.image_dir / self.dataset_gesture_name # create dir for new shape
            filepath.mkdir(exist_ok=True)

    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        predictions = self.model.predict(image)
        _, predicted_class_index = torch.max(predictions, 1)
        predicted_class = ModelConfig.classes[predicted_class_index.item()]
        print(f"Predicted Class : {predicted_class}")
        return predicted_class

    def on_press(self, key):
        state = queue.get()
        if key == GestureRecognitionSettings.gesture_hot_key:
            state.gesture_capture = True
            log.debug('Gesture capture enabled')
            state.current_mouse_sequence = []

        queue.put(state)

    def on_release(self, key):
        state = queue.get()
        if key == GestureRecognitionSettings.gesture_hot_key:
            state.gesture_capture = False
            log.debug('Gesture capture disabled')
            state.last_mouse_sequence = state.current_mouse_sequence
            log.debug(f'coordinates = {state.last_mouse_sequence}')
            if self.mode == Mode.LIVE:
                filepath = GestureRecognitionSettings.live_image_dir
                filepath = save_drawing(state.last_mouse_sequence, directory=filepath)
                predicted_gesture = self.predict_image(filepath)
                log.info(f"Predicted action : {predicted_gesture}")
                self.command_executor.run_command(predicted_gesture)
            elif self.mode == Mode.CREATE_DATASET:
                filepath = GestureRecognitionSettings.image_dir / self.dataset_gesture_name
                filepath = save_drawing(state.last_mouse_sequence, directory=filepath)

            state.last_mouse_sequence = []
            state.current_mouse_sequence = []
        queue.put(state)

    def on_move(self, x, y):
        state = queue.get()
        # print('Pointer moved to {0}'.format((x, y)))
        if state.gesture_capture:
            state.current_mouse_sequence.append((x, y))
        queue.put(state)

    def initialise_workspace(self):
        GestureRecognitionSettings.image_dir.mkdir(exist_ok=True)

    def configure_input_listeners(self):
        keyboard_listener = KeyboardListener(on_press=self.on_press, on_release=self.on_release)
        mouse_listener = MouseListener(on_move=self.on_move)
        return keyboard_listener, mouse_listener

    def start(self):
        log.info("Hold ctrl key and use mouse pointer to make a gesture, then release ctrl key")
        self.initialise_workspace()

        keyboard_listener, mouse_listener = self.configure_input_listeners()

        # Start the threads and join them so the script doesn't end early
        keyboard_listener.start()
        mouse_listener.start()
        keyboard_listener.join()
        mouse_listener.join()


if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--create-dataset', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.start:
        gre = GestureRecognitionEngine(mode=Mode.LIVE)
        gre.start()
    elif args.create_dataset:
        gre = GestureRecognitionEngine(mode=Mode.CREATE_DATASET)
        gre.start()
    elif args.train:
        print('Training Shape Detector Model..')
        clean_old_train_test_directories()
        copy_to_train_test_subdirectories()
        start_training()
    else:
        print('Check the usage. \n\n '
              'For Starting the program : \n'
              'python3 gesture_recognition_engine.py --start\n\n'
              'For creating dataset for training \n'
              'python3 gesture_recognition_engine.py --create-dataset')


