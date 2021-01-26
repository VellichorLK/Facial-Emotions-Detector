# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:23:34 2018

@author: laura
"""


import pickle
import numpy as np
import emotions_cam

emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]


with open('models/trained_svm_model', 'rb') as f:
   model = pickle.load(f)



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
def emotions(capture):
        
        ret, frame = capture.read()
        emotions_cam.draw_landmarks(frame)
        faces = np.array([emotions_cam.get_landmarks(frame)])
        if faces[0] is not None:
            prediction = model.predict_proba(faces)
            if len(prediction) > 0:
                return prediction
       
            

            