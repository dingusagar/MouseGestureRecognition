'''
Script to capture the mouse coordinates on press of a control key.
'''


from pynput.keyboard import Key
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener


class cfg:
    gesture_hot_key = Key.ctrl_l

class InteractionState:
    gesture_capture = False
    last_mouse_sequence = []
    current_mouse_sequence = []


state = InteractionState()


def on_press(key):
    global state
    if key == cfg.gesture_hot_key:
        state.gesture_capture = True
        # state.last_mouse_sequence = []
        # state.current_mouse_sequence.append(pyautogui.position())
        print('Gesture capture enabled')
        return True

def on_release(key):
    global state
    if key == cfg.gesture_hot_key:
        state.gesture_capture = False
        # print('Gesture capture disabled')
        state.last_mouse_sequence = state.current_mouse_sequence
        state.current_mouse_sequence = []
        print(f'Captured sequence : {state.last_mouse_sequence}')




def on_move(x, y):
    global state
    # print('Pointer moved to {0}'.format((x, y)))
    if state.gesture_capture:
        state.current_mouse_sequence.append((x,y))


keyboard_listener = KeyboardListener(on_press=on_press, on_release=on_release)
mouse_listener = MouseListener(on_move=on_move)

# Start the threads and join them so the script doesn't end early
keyboard_listener.start()
mouse_listener.start()
keyboard_listener.join()
mouse_listener.join()