# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:09:39 2017

@author: laura
"""


import cv2
import dlib
import numpy as np
import math
from sklearn.svm import SVC
import glob
import random
import pickle

movements = ["neutral","blink","left","right"]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel = 'linear',probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []

def get_files(movement): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset1\\%s\\*" %movement)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def get_landmarks(image):
    faces = detector(image, 1)
    for i,face in enumerate(faces): #For all detected face instances individually
        shape = predictor(image, face) #Draw Facial Landmarks with the predictor class
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

        data['landmarks_vectorised'] = landmarks_vectorised
        
    if len(faces) < 1: 
        data['landmarks_vestorised'] = "error"
        
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for movement in movements:
        print(" working on %s" %movement)
        training, prediction = get_files(movement)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(movements.index(movement))
    
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(movements.index(movement))

    return training_data, training_labels, prediction_data, prediction_labels   


accur_lin = []
for i in range(0,5):
    print("Making sets %s" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)
    
    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print ("linear: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list
print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs
with open('models/trained_expressions_model', 'wb') as f:
        pickle.dump(clf, f)
