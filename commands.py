import subprocess


class CLICommand:
    def __init__(self, command, description=None, wait_for_exit=False):
        self.command = command
        self.description = description
        self.wait_for_exit = wait_for_exit

    def execute(self, ):
        print(f"Executing Command - {self.description}")
        if not self.wait_for_exit:
            subprocess.Popen([self.command],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True)
        else:
            subprocess.run(self.command.split())


class PythonBasedCommand:
    def __init__(self, python_func, description=None):
        self.python_func = python_func
        self.description = description

    def execute(self):
        print(f"Executing Command - {self.description}")
        self.python_func()