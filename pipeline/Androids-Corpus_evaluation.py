#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  1/9/23 8:47

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import os
os.chdir("../")

import numpy as np
import random
import h5py
from config.config import FEATURE_PARMS


def making_sequence(features, timestep = None):
    size = features.shape[0]
    feature_dim = features.shape[1]
    if size < timestep:
        pad = np.zeros((timestep - size, feature_dim))
        features = np.concatenate((features,pad))
        size = timestep
        # raise ValueError("Sample shorter than minimun timestep")
    div = size // timestep
    size_new = int(div * timestep)
    features = features[:size_new]
    features = features.reshape((div, timestep, feature_dim))

    return  features
def batch_elements(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i+batch_size]

TASK_LIST = ["Interview-Task", "Reading-Task"]  #
CORPUS = "Androids-Corpus"
feature_type = "melSpectrum"
TIMESTEP = FEATURE_PARMS[feature_type][1]

for task in TASK_LIST:
    label = np.genfromtxt(f"label/label_{CORPUS}_{task}.txt", dtype=str, delimiter=" ")[1:,:]
    folds = np.genfromtxt(f"default-folds_Androids-Corpus/fold_{task}.txt", delimiter=",",dtype=str)
    hf = h5py.File(f"features/{feature_type}_{CORPUS}_{task}.h5", 'r')

    folds_test = []
    folds_train = [[],[],[],[],[]]
    x_train =[]
    y_train = []
    for i in range(folds.shape[0]):
        folds_test.append(folds[i].split(" "))
        for j, name in enumerate(label[:,0]):
            if not name in folds[i]:
                folds_train[i].append(name)
                feat = making_sequence(np.asarray(hf[name]), timestep = TIMESTEP)
                x_train.append(feat)
                y_train.append([label[j,-1].astype(float)]*feat.shape[0])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    index = list(range(x_train.shape[0]))
    random.shuffle(index)
    batch_size = 128

    # TODO Initialize model

    # TODO Training loop

    for batch in batch_elements(index, batch_size):
        print(batch)



