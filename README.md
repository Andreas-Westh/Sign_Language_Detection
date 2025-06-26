# Sign Language Detection (Letters)

Just a little project where I train a model to recognize sign language letters using webcam input and keypoint data (from MediaPipe). 
Right now it just detects individual letters like A, B, etc. Plan is to make it smarter over time so it can actually spell out words.
Currently training it on Danish Sign Language made by myself, as I haven't been able to find a suitible dataset for Danish Sign Language to train on.

---

### Files:

- `image_collection.py`  
  Opens up the webcam so I can collect training images for each letter. Saves the keypoints and labels (the actual letter the sign reprecents).

- `data_creation.py`  
  Converts the saved images/keypoints into proper features and labels (x, y) for model training.

- `train_RF.py`  
  Trains a Random Forest classifier on the data. Saves the model when done.

- `test_classifier.py`  
  Loads the model and runs live detection through the webcam. Shows the predicted letter in real-time.

---

### Notes:
- Right now it’s just for static signs (like A, B, C…).
- Plan is to add logic for spacing, word detection, and support for Danish letters like Æ, Ø, Å (maybe using motion).
- Very much a work in progress.
