import cv2
import numpy as np
import mediapipe as mp

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import models,layers
from tensorflow.keras import optimizers

#Loading model
model = load_model('asl_alphabet.h5')
#total predictable classes
classes = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','del','space','Nothing']
cur_text = ''

font = cv2.FONT_HERSHEY_SIMPLEX

#using mediapipe for hands detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.4)

cap = cv2.VideoCapture(0)

while(True):
    #height and width of the window
    height = int(cap.get(4))
    width = int(cap.get(3))

    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        print("ignore empty cam frame")
        continue

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #results contain the data of the recorded hand
    results = hands.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        hand_pts=[]
        f_hand = results.multi_hand_landmarks[0]

        min_x=float('inf')
        min_y=float('inf')
        max_x=float('-inf')
        max_y=float('-inf')
        
        for id, lm in enumerate(f_hand.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            min_x, min_y = min(min_x, cx), min(min_y, cy)
            max_x, max_y = max(max_x, cx), max(max_y, cy)
            
            # cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            hand_pts.append([cx,cy])
    else:    
        cv2.putText(frame,"No hand found in frame", (width//2, height//2), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        continue

    min_x,min_y = min_x-30, min_y-30
    max_x,max_y = max_x+30, max_y+30

    min_x = int(max(0, min_x))
    min_y = int(max(0, min_y))
    max_x = int(min(width, max_x))
    max_y = int(min(height, max_y)) 

    cv2.rectangle(frame,(min_x, min_y),(max_x, max_y),(255,255,255), 2)

    #cropped frame contains the part only containing the hand
    cropped_frame = frame[min_y: max_y, min_x: max_x]

    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
    resized_frame = cv2.resize(cropped_frame, (200,200))
    reshaped_frame = (np.array(resized_frame)).reshape((1,200,200,3))
    frame_for_model = reshaped_frame / 255

    #predicting using the cropped image
    prediction = np.array(model.predict(frame_for_model))
    predicted_class = classes[prediction.argmax()]  

    prediction_probability = prediction[0, prediction.argmax()]

    if (prediction_probability > 0.2) and results.multi_hand_landmarks:
        cv2.putText(frame, f' {predicted_class} - {( prediction_probability * 100):.2f}%', 
                                (min_x + 10, max_y+5), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
    else:
        predicted_class=' '
        cv2.putText(frame, f' space - {( prediction_probability * 100):.2f}%', 
                                (min_x + 10, max_y+5), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        #entring the character in cur_text on pressing s
        alph = predicted_class #chr(num+96)
        cur_text += alph
        print(cur_text)
    #printing current characters
    cv2.rectangle(frame,(width-320, height-40),(width,height),(0,0,0), -1)
    cv2.putText(frame,f"Text:- {cur_text}", (width+10, height-22), font, 1, (255,255,255), 2, cv2.LINE_AA)

    # Display the resulting frameq
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(cur_text)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()