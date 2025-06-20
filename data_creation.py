import os

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# landmark drawings for images, not needed for classification but looks rad
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# raw data folder, where image_collection.py saves raw images
DATA_DIR = './data'

# arrays for saving hand landmarks 
data = []
labels = []


for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # skip .DS_Store or anything weird
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): #[:1] can be added before the : to only itterete through one image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # mediapipe uses rgb, so we need to convert the raw images so landmarks can work
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # detects landmarks in the image
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # itterates through results and plots the landmarks onto them
                            # for drawing on the image, mainly for testing
            for hand_landmarks in results.multi_hand_landmarks:
                # for drawing on the image, mainly for testing
                #mp_drawing.draw_landmarks(
                #    img_rgb, # image to draw
                #    hand_landmarks, #model output
                #    mp_hands.HAND_CONNECTIONS,
                #    mp_drawing_styles.get_default_hand_landmarks_style(),
                #    mp_drawing_styles.get_default_hand_connections_style())
                for i in range(len(hand_landmarks.landmark)):
                    #print(hand_landmarks.landmark[i]) # simple print of landmarks, only x and y needed
                    # these values will be out into an array, meaning we only process the actual landmarks, not the whole image
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                