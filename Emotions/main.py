# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:57:36 2017

@author: laura
"""
import pickle
import cv2
import numpy as np
import emotions_cam


emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


with open('models/trained_svm_model', 'rb') as f:
   model = pickle.load(f)

with open('models/trained_MLP_model', 'rb') as f:
   model1 = pickle.load(f)


print('kek')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

for frame in emotions_cam.get_webcam_video(350,350):
    emotions_cam.draw_landmarks(frame)
    faces = np.array([emotions_cam.get_landmarks(frame)])
    
    if faces[0] is not None:
        
        prediction = model.predict(faces)
        prediction1 = model1.predict(faces)
        if len(prediction) > 0:
            text = emotions[prediction[0]]
            cv2.putText(frame, "SVM", (40, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 0, 255),
                      thickness=2)
            cv2.putText(frame, text, (40, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                      thickness=2)
        if len(prediction1) > 0:
            text = emotions[prediction1[0]]
            cv2.putText(frame, "MLP", (200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 0, 255),
                      thickness=2)
            cv2.putText(frame, text, (200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                      thickness=2)
           
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break  
    cv2

cv2.destroyAllWindows()