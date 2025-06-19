import cv2
import os
import time

# raw data folder for images
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

# open cam (0 = either iphone or monitor, depends on what my mac feels like it seems)
cap = cv2.VideoCapture(0)  # change index if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# number of signs and pics per
classes = 3
images_per_class = 100

for label in range(classes):
    path = os.path.join(DATA_DIR, str(label))
    os.makedirs(path, exist_ok=True)

    # just a simple wait till q is pressed to begin
    while True:
        _, frame = cap.read()
        cv2.putText(frame, 'Press q to start, esc to quit', (80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
    # simple 3 sec countdown when q is pressed
    for i in range(3, 0, -1):
        _, frame = cap.read()
        cv2.putText(frame, f'{i}', (250, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)

    # takes images
    for i in range(images_per_class):
        _, frame = cap.read()
        cv2.putText(frame, f'{i}/{images_per_class}', (30, 50), # these 2 lines are live image within class counter
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(path, f'{i}.jpg'), frame)
        
        # exits if esc is pressed
        if cv2.waitKey(25) == 27:  # ESC to quit anytime
            cap.release()
            cv2.destroyAllWindows()
            exit()

# clean up when done
cap.release()
cv2.destroyAllWindows()

