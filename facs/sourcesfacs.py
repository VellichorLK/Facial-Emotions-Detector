# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:36:53 2018

@author: laura
"""

import glob
import shutil

facs = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","54","61","62","63","64"] #Define FACS
participants = glob.glob("source_FACS\\*") #Returns a list of all folders with participant numbers
for x in participants:
    part = "%s" %x[-4:] #store current participant number
    for sessions in glob.glob("%s\\*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s\\*" %sessions):
            current_session = files[17:-27]
            
            file = open(files, 'r')
            AUs = file.readlines()#emotions are encoded as a float, readline as float, then convert to integer.
            for lines in AUs:
                line=lines.split("   ")
                AU=int(float(line[1]))
                sourcefile_neutral = glob.glob("source_images\\%s\\%s\\*" %(part, current_session))[0] #do same for neutral image
                sourcefile_fac = glob.glob("source_images\\%s\\%s\\*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
                print(sourcefile_fac)
                dest_neut = "sorted_set_facs\\neutral\\%s" %sourcefile_neutral[25:] #Generate path to put neutral image
                dest_fac = "sorted_set_facs\\%s\\%s" %(str(AU), sourcefile_fac[25:]) #Do same for emotion containing image
                shutil.copyfile(sourcefile_neutral, dest_neut) #Copy file
                shutil.copyfile(sourcefile_fac, dest_fac) #Copy file
                #print("ok")