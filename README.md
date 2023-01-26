# MouseGestureRecognition
An experimental project idea to see if we can make gestures with the mouse cursor and detect those gestures to perform certain actions.



# Initial Idea : 
* A process that captures the mouse coordinates continously. (or better idea is to trigger the mouse capture on a key board shortcut like ctrl)
* Transform these absolute coordinate sequence to a relative sequence. so that the position in the screen where this gesture was done does't matter. 
* Train in offline a sequence to sequence model like transformer or LSTM encode a sequence. 
* At inference time, a sequence can be passed on to this model and we should get a embedding vector for that sequence. 
* Use vector search to match if the current sequence vector matches closely with any existing vector described at the setup time. 
* Trigger the action corresponding to the sequence. 


### Update 26 Nov:
  * Can we solve this problem using CNN. capture the mouse coordinates and treat it as an image and pass it to a cnn ?
  * Capture mouse cooordinates works. Now need to figure out how to transform this into an image. 

# Explorations: 
* How to transform absolute coordinates to relative coordinates. 
* Which sequence model works with minimal training data. Any pretrained models available publicly ?



# Version 1 Progress
### Tweaked the idea little bit and simplified it. 
* Simplified the whole problem by re-thinking it as a image classification instead of sequence classification of x,y coordinates.
* What we can do is, capture the x,y coordinates of the mouse movements after pressing a hot key like ctrl. 
* After this, plot these x,y points in as a scatter plot in matplotlib, we can connect the points to get a smooth figure, remove the axis and save the figure. 
* Now we have simplified the problem to a image classification problem. 
* We can make use of a variety of lightweight CNN architectures like mobilenet, resnet18 etc to predict the image, there by predicting the gesture.
* Once we have this shape classifier, we can associate various commands either shell commands or python programs to be run when this gesture is predicted. 


### How to run the program. 
To test run the program, do this.
```py
python3 gesture_recognition_engine.py --start
```

The current model has 3 gestures trained - circle, rectangle, triangle. 
So you can hold the hot key (ctrl key) and use the mouse curser to draw any of these gestures. 
On release of the hot key, the model will predict the gesture and execute the corresponding command. 
All gesture to command mappings are present in gesture_mapping.py

For example, a circle gesture will open google.com in a new tab in the chrome browser. 
rectangle gesture will open calculator.

These commands are tested on ubuntu system. you could take a look at the gesture_mapping.py and modify the commands to your needs. 
You could also attach custom python functions to be executed on the detection of these gestures. Check gesture_mapping.py for more details.


### How to add new gestures. 

First we need to create a dataset of few images with the new gesture. 
for that, run the following

```py
python3 gesture_recognition_engine.py --create-dataset
```

Now you can hold the hotkey and create a gesture. it will be saved in the images folder. 
keep doing different variations of this. around 10 to 15 images should be enough. 

Now train the model using this command. 
```py
python3 gesture_recognition_engine.py --train
```

once this model is trained you can use this command to start the gesture recognition program. 

```py
python3 gesture_recognition_engine.py --start
```

All configuration and settings are present in cfg.py. You can change the path to the correct model checkpoint 
using ModelConfig.latest_checkpoint_path


# Version 2 - TODO
* Make inference much faster by using a more lightweight cnn model. We just need shape recognition. 
* Reduce memory footprint of the model. 
* a new shape/gesture requires around 10-15 images. Even though this is small, its still a manual process. Why cant we think of encoding the image and doing a vector similarity, this way we can make it few shot for new classes. 
