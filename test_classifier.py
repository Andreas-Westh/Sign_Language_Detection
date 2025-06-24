import cv2
import mediapipe as mp
import pickle 
import numpy as np

# load model the given model 
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

webcam_capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    
    data_aux = []
    x_ = []
    y_ = []
    
    ret, frame = webcam_capture.read()
    if frame is None: continue
    
    # raw landmark values is float, wanna make into int
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # detects landmarks in the frame
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
                #model visualisation
                mp_drawing.draw_landmarks(
                    frame, # frame to draw
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # itterating theough landmarks
        for h in range(2):  # max 2 hands
            if h < len(results.multi_hand_landmarks):
                hand_landmarks = results.multi_hand_landmarks[h]
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
            else:
                data_aux.extend([0.0] * 42)  # pad if hand missing

        # values for visual bounding box (for corners)
        x1 = int(min(x_) * W) - 12
        y1 = int(min(y_) * H) - 12
        x2 = int(max(x_) * W) - 12 
        y2 = int(max(y_) * H) - 12
        
                    
        # use classification model
        prediction = model.predict([np.asarray(data_aux)])
        
        # predicted character
        predicted_character = prediction[0]
        
        # live visual of prediction 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        

                
    cv2.imshow('frame', frame)
    cv2.waitKey(1) # waits x milisecodns between each frame



webcam_capture()
cv2.destroyAllWindows()