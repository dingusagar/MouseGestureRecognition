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



