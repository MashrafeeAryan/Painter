import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warning, 3 = error only

import cv2
import mediapipe as mp
video = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands()

while True:
    _, frame = video.read()
    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    landmark = hands.process(rgb_frame)
    if landmark.multi_hand_landmarks:
        for hand in landmark.multi_hand_landmarks:
            index_tip = hand.landmark[8]
            if (index_tip):
                print("Index Tip")
    else:
        print("No Hands")
    cv2.imshow("Webcam", flipped_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
