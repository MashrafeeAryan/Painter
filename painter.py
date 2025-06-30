import os

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warning, 3 = error only

import cv2
import mediapipe as mp
video = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands()
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.imshow("Canvas", canvas)
prev_x, prev_y = None, None
while True:
    _, frame = video.read()
    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    landmark = hands.process(rgb_frame)
    if landmark.multi_hand_landmarks:
        for hand in landmark.multi_hand_landmarks:
            index_tip = hand.landmark[8]
            x = int(index_tip.x * 640)
            y = int(index_tip.y * 480)
            prev_x, prev_y = x, y
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0,255,0), thickness=3)
    else:
        print("No Hands")
        prev_x, prev_y = None, None
    cv2.imshow("Webcam", flipped_frame)
    cv2.imshow("Canvas", canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


