from gesture_mapping import GestureMapping
class CommandExecutor:
    def __init__(self, gesture_callbacks):
        self.gesture_callbacks = gesture_callbacks

    @classmethod
    def from_gesture_mapping(cls, gesture_mapping):
        return cls(gesture_mapping)

    def run_command(self, gesture):
        command = self.gesture_callbacks.get(gesture, None)
        if command:
            command.execute()

if __name__ == "__main__":
    command_executor = CommandExecutor.from_gesture_mapping(GestureMapping)
    command_executor.run_command('rectangle')