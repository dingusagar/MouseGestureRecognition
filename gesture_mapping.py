from commands import CLICommand, PythonBasedCommand


GestureMapping = {
    'rectangle': CLICommand("gnome-calculator", description="Open Calculator"),
    'circle': CLICommand("google-chrome https://www.google.com/", description="Open google in chrome"),
    'triangle': CLICommand("xdg-open /media/dingusagar/Data/", description="Open Data Folder")
}

