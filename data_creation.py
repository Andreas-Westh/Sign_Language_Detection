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

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # skip .DS_Store or anything weird
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # mediapipe uses rgb, so we need to convert the raw images so landmarks can work
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # detects landmarks in the image
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # itterates through results and plots the landmarks onto them
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb, # image to draw
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        plt.figure()
        plt.imshow(img_rgb)

plt.show()