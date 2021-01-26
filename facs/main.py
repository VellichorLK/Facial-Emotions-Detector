# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:29:53 2018

@author: laura
"""


import pickle
import cv2
import numpy as np
import emotions_cam


facs1 = ["inner brow raise", "neutral"]
facs2 = ["outer brow raiser","neutral"]
facs4 = ["brow lowerer", "neutral"]
facs5 = ["upper lid raise","neutral"]
facs6 = ["Cheek raise","neutral"] 
facs7 = ["lid tighten", "neutral"]
facs9 = ["nose wrinkle", "neutral"]
facs10 = ["upper lip raise", "neutral"]
facs12 = ["Lip corner pull","neutral"]
facs14 = ["dimple", "neutral"]
facs15 = ["lip corner depress", "neutral"]
facs17 = ["chin raise", "neutral"]
facs20 = ["lip strech", "neutral"]
facs23 = ["lip tighten", "neutral"]
facs24 = ["lip press", "neutral"]
facs25 = ["lips part","neutral"]
facs26 = ["jaw drop", "neutral"]

with open('models/trained_1_model', 'rb') as f:
    model1 = pickle.load(f)
   
with open('models/trained_2_model', 'rb') as f1:
    model2 = pickle.load(f1)

with open('models/trained_4_model', 'rb') as f2:
    model4 = pickle.load(f2)
   
with open('models/trained_5_model', 'rb') as f3:
    model5 = pickle.load(f3)
   
with open('models/trained_6_model', 'rb') as f4:
    model6 = pickle.load(f4)

with open('models/trained_7_model', 'rb') as f5:
    model7 = pickle.load(f5)

with open('models/trained_9_model', 'rb') as f6:
    model9 = pickle.load(f6)

with open('models/trained_10_model', 'rb') as f7:
    model10 = pickle.load(f7)
   
with open('models/trained_12_model', 'rb') as f8:
    model12 = pickle.load(f8)
   
with open('models/trained_14_model', 'rb') as f9:
    model14 = pickle.load(f9)

with open('models/trained_15_model', 'rb') as f10:
    model15 = pickle.load(f10)

with open('models/trained_17_model', 'rb') as f11:
    model17 = pickle.load(f11)

with open('models/trained_20_model', 'rb') as f12:
    model20 = pickle.load(f12)

with open('models/trained_23_model', 'rb') as f13:
    model23 = pickle.load(f13)

with open('models/trained_24_model', 'rb') as f14:
    model24 = pickle.load(f14)
   
with open('models/trained_25_model', 'rb') as f15:
    model25 = pickle.load(f15)

with open('models/trained_26_model', 'rb') as f16:
    model26 = pickle.load(f16)

print('kek')

i  = 0                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
for frame in emotions_cam.get_webcam_video(350,350):
    print ('punto 1')
    if (i%48 == 0):
        print('ENTO: ',i)
        emotions_cam.draw_landmarks(frame)
        faces = np.array([emotions_cam.get_landmarks(frame)])
        eyes = np.array([emotions_cam.get_landmarks_eyes(frame)])
        browseyes = np.array([emotions_cam.get_landmarks_browseyes(frame)])
        mouth1 = np.array([emotions_cam.get_landmarks_mouth(frame)])
        mouth = np.array([emotions_cam.get_landmarks_mouth1(frame)])
        brows1 = np.array([emotions_cam.get_landmarks_brows(frame)])
        brows = np.array([emotions_cam.get_landmarks_brows2(frame)])
        nose = np.array([emotions_cam.get_landmarks_nose(frame)])
        jaws = np.array([emotions_cam.get_landmarks_jaws(frame)])
        if faces[0] is not None:
            pred_1 = model1.predict(brows1)
            pred_2 = model2.predict(brows1)
            pred_4 = model4.predict(faces)
            pred_5 = model5.predict(browseyes)
            pred_6 = model6.predict(faces)
            #pred_7 = model7.predict(eyes)
            pred_9 = model9.predict(nose)
            pred_10 = model10.predict(mouth)
            pred_12 = model12.predict(mouth1)
            pred_14 = model14.predict(mouth1)
            pred_15 = model15.predict(mouth)
            pred_17 = model17.predict(mouth)
            pred_20 = model20.predict(mouth)
            pred_23 = model23.predict(mouth1)
            pred_24 = model24.predict(mouth)
            pred_25 = model25.predict(mouth)
            pred_26 = model26.predict(mouth)
            
            if (len(pred_1)> 0) & (len(pred_2)> 0) & (len(pred_4)> 0) & (len(pred_5)> 0) & (len(pred_6)> 0) & (len(pred_9)> 0) & (len(pred_10)> 0) & (len(pred_12)>0) & (len(pred_14)> 0) & (len(pred_15)> 0) & (len(pred_17)> 0) & (len(pred_20)> 0) & (len(pred_23)> 0) & (len(pred_24)> 0) & (len(pred_25)>0) & (len(pred_26)> 0):
                
                if (pred_23 ==0) & (pred_24 == 0) & (pred_4 == 0):
                    cv2.putText(frame,  "Angry", (150,220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs4[pred_4[0]], (220, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs23[pred_23[0]],(220, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs24[pred_24[0]],(220, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                
                elif (pred_1 == 0) & (pred_2 == 0) & (pred_25 == 0) & (pred_26 ==1):
                    cv2.putText(frame,  "Fear", (150,220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs1[pred_1[0]], (220, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs2[pred_2[0]],(220, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs25[pred_25[0]],(220, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    
                elif (pred_1 == 0) & (pred_4 == 0) & (pred_15 ==0) :
                    cv2.putText(frame,  "Sadness", (150,220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs4[pred_4[0]], (220, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs1[pred_1[0]],(220, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs15[pred_15[0]],(220, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    
                elif (pred_5 == 0) & (pred_26 == 0):
                    cv2.putText(frame,  "Surprise", (150,220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs5[pred_5[0]], (220, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs26[pred_26[0]],(220, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    
                
                elif (pred_6 == 0) & (pred_12 == 0) & (pred_25 == 0):
                    cv2.putText(frame,  "Happy", (150,220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs6[pred_6[0]], (220, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs12[pred_12[0]],(220, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs25[pred_25[0]],(220, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    
                elif (pred_14 == 0):
                    cv2.putText(frame,  "Contempt", (150,220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs14[pred_14[0]], (220, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    
                    
                elif (pred_9 == 0) & (pred_10 == 0) & (pred_25 == 0) :
                    cv2.putText(frame,  "Disgust", (150,220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs9[pred_9[0]], (220, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs10[pred_10[0]],(220, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                    cv2.putText(frame,  facs25[pred_25[0]],(220, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
                else:
                    cv2.putText(frame,  "Neutral", (150,220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                          thickness=1)
        cv2.imshow('SVM', frame)
    i = i + 1
    
    
    if cv2.waitKey(1) & 0xFF == 27:
        break  
 
cv2.destroyAllWindows()