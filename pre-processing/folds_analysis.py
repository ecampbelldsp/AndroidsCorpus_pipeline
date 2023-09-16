#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  16/9/23 11:27

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""

import os
os.chdir("../")

import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  #TKAgg
import soundfile as snd





if __name__ == "__main__":


    #Picking GPU
    TASK_LIST = ["Reading-Task","Interview-Task" ]  # , "Reading-Task"
    CORPUS = "Androids-Corpus"
    root_audio = "/media/ecampbell/D/Data-io/Androids-Corpus/"


    for task in TASK_LIST:


        # Loading default distribution folds files
        folds = np.genfromtxt(f"default-folds_Androids-Corpus/fold_{task}.txt", delimiter=",", dtype=str)
        num_folds = folds.shape[0]

        ID = f"{datetime.datetime.now()}"

        label = np.genfromtxt(f"label/label_{CORPUS}_{task}.txt", dtype=str, delimiter=" ")[1:,:]

        #Creating K-Folds and loading features
        folds_test = []
        with open(f"Corpora_analysis/{CORPUS}/{task}folds_descriptions","w") as file:
            file.write("Fold Age(>=40) Age (<40) Female Male Average-Recording-Duration(min) Low-education High-education Depressed Non-depressed\n")
            for f in range(folds.shape[0]):

                duration = []
                gender = []
                age = []
                educational_level = []
                depression = []
                for j, name in enumerate(label[:,0]):
                    if  name in folds[f]:
                        samples, fs = snd.read(f"{root_audio}{task}/audio_clip_gathering/{name}")
                        duration.append(len(samples)/(fs*60))
                        gender.append(label[j,2])
                        age.append(float(label[j,4]))
                        try:
                            educational_level.append(float(label[j,6]))
                        except ValueError:
                            ...
                        depression.append(float(label[j,-1]))
                    else:
                        continue
                age_higher_40 = len(np.where(np.asarray(age) >= 40)[0])
                age_lower_40 = len(np.where(np.asarray(age) < 40)[0])
                female = len(np.where(np.asarray(gender) == "F")[0])
                male = len(np.where(np.asarray(gender) == "M")[0])
                duration_average = round(np.mean(duration),2)
                edu_high = len(np.where(np.asarray(educational_level)>=3)[0])
                edu_low = len(np.where(np.asarray(educational_level)<3)[0])
                depressed = len(np.where(np.asarray(depression) == 1)[0])
                non_depressed = len(np.where(np.asarray(depression) == 0)[0])

                file.write(f"{f} {age_higher_40} {age_lower_40} {female} {male} {duration_average} {edu_low} {edu_high} {depressed} {non_depressed}\n\n")