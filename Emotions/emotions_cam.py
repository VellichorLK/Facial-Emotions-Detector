# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:34:04 2017

@author: laura
"""

import cv2
import numpy as np
import math
import dlib


detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_webcam_video(width, height):
    vc = cv2.VideoCapture(0)
    vc.set(3, width)
    vc.set(4, height)
    print(vc.isOpened())

    while True:
        ret, frame = vc.read()
       
        yield frame

def detection(image):
    faces = detector(handle(image),1)
    return faces[0] if len(faces)>0 else None

def resized_frame(image):
    if detection(image) is not None:
        resized = resize_face(image,detection(image))
    else:
        resized = None
    return resized

def resize_detection(image):
    if resized_frame(image) is not None:
        re_detection = detector(resized_frame(image),1)
    else:
        re_detection = []
    return re_detection[0] if len(re_detection)>0 else None

def resize_face(image,rect):
    clahe_image = handle(image)
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    cut = clahe_image[y:y+h,x:x+w]
    try:
            return cv2.resize(cut, (350, 350))
    except:
            return None

def draw_landmarks(frame):
    clahe_image = handle(frame)
    faces = detector(clahe_image, 1) #Detect the faces in the image

    for i,face in enumerate(faces): #For each detected face
        
        shape = predictor(clahe_image, face) #Get coordinates
        for k in range(68): 
            cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (0,0,255), -1) #For each point, draw a red circle with thickness2 on the original frame


def get_landmarks(image):
            
        frame = resized_frame(image)
        rect = resize_detection(image)
        if rect is None:
            return None
        #(x, y, w, h) = face_utils.rect_to_bb(rect)

        #shape = predictor(frame, rect) #Draw Facial Landmarks with the predictor class
        shape = predictor(frame,rect)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        xshape = []
        yshape = []
        for k in range(68): #Store X and Y coordinates in two lists
            xshape.append(float(shape.part(k).x))
            yshape.append(float(shape.part(k).y))
        xmean = np.mean(xshape) #Find both coordinates of centre of gravity
        ymean = np.mean(yshape)
        xcentral = [(x-xmean) for x in xshape] #Calculate distance centre <-> other points in both axes
        ycentral = [(y-ymean) for y in yshape]
        
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xshape, yshape): # paara hacer pares 
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean)) #convert input to an array
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        return landmarks_vectorised


def handle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    return clahe_image
  


    

