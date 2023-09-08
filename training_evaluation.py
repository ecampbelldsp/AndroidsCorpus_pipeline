#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  1/9/23 8:47

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import gc
import os

import datetime
import numpy as np
import matplotlib
# matplotlib.use('qtagg')
import matplotlib.pyplot as plt

import h5py
import torch

from mlflow import log_metric,log_metrics, log_param, log_params, log_artifact
import mlflow

from config.config import FEATURE_PARMS
from ai import DeepAudioNet
from ai.training import train_CNN_LSTM
from box.utilities import computing_performance, making_sequence, split_train_data, select_gpu_with_most_free_memory,heat_map
from box.utilities import plot_loss


if __name__ == "__main__":
    #Picking GPU
    selected_GPU, device = select_gpu_with_most_free_memory()
    RUNS = 2
    TASK_LIST = ["Interview-Task"]  # , "Reading-Task"
    CORPUS = "Androids-Corpus"
    FEATURE_TYPE_LIST = ["melSpectrum", "rasta", "compare_lld","egemap_lld", "hubert_base", "wav2vec2_base"]
    with_interviewer = False
    # feature_type = "melSpectrum"#"melSpectrum" compare_lld egemap_lld rasta hubert_base wav2vec2_base
    for feature_type in FEATURE_TYPE_LIST:
        if not os.path.exists("model"):
            os.mkdir("model")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        for task in TASK_LIST:
            #Setting Mlflow experiment name
            mlflow.set_experiment(f"{CORPUS}-{task}")
            # Define hyperparameters

            batch_size = 128
            learning_rate = 0.001
            num_epochs = 10
            validation_rate = 0.3
            EARLY_STOP = {"Do":True, "max_attempts": 10}
            TIMESTEP = FEATURE_PARMS[feature_type][1]

            params = {
                    "batch_size": batch_size, "learning_rate": learning_rate,
                    "Epochs": num_epochs,"Early_stop": EARLY_STOP,
                    "validation_rate": validation_rate, "runs":RUNS,
                    "Feature": feature_type,"Sequence length":TIMESTEP, "device":device,
                    "Interviewer in recording": with_interviewer
            }
            # Loading default distribution folds files
            folds = np.genfromtxt(f"default-folds_Androids-Corpus/fold_{task}.txt", delimiter=",", dtype=str)
            num_folds = folds.shape[0]
            metric_fold = {"accuracy": np.zeros((RUNS,num_folds)), "recall": np.zeros((RUNS,num_folds)),
                           "precision": np.zeros((RUNS,5)), "auc": np.zeros((RUNS,num_folds)), "f1": np.zeros((RUNS,num_folds))}

            ID = f"{datetime.datetime.now()}"
            with mlflow.start_run(run_name=f"{feature_type}_{ID}") as run:
                log_params(params)
                for r in range(RUNS):


                    label = np.genfromtxt(f"label/label_{CORPUS}_{task}.txt", dtype=str, delimiter=" ")[1:,:]


                    if not with_interviewer:
                        hf = h5py.File(f"features/{feature_type}_{CORPUS}_{task}.h5", 'r')
                    else:
                        hf = h5py.File(f"features/{feature_type}_{CORPUS}_{task}_with_interviewer.h5", 'r')
                    #Creating K-Folds and loading features
                    folds_test = []
                    for f in range(folds.shape[0]):
                        x_train = []
                        y_train = []
                        folds_test.append(folds[f].split(" "))
                        for j, name in enumerate(label[:,0]):
                            if not name in folds[f]:
                                feat = making_sequence(np.asarray(hf[name]), timestep = TIMESTEP)#[:20,:,:]
                                x_train.append(feat)
                                y_train.append([label[j,-1].astype(float)]*feat.shape[0])

                        print("Loading training data (numpy)")
                        x_train,y_train,x_val,y_val = split_train_data(x_train, y_train, validation_rate)

                        print("Creating Dataloader")
                        x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
                        y_train = torch.from_numpy(y_train).type(torch.FloatTensor)

                        x_val = torch.from_numpy(x_val).type(torch.FloatTensor)
                        y_val = torch.from_numpy(y_val).type(torch.FloatTensor)

                        print("Model initialization")
                        model = DeepAudioNet.CustomMel1(in_channels=x_train.shape[1], outputs=1).to(device)
                        model,loss = train_CNN_LSTM(model,params, train = [x_train,y_train], val = [x_val, y_val ], r = r, f = f)

                        path_png, path_pdf = plot_loss(loss[0], loss[1], R=r, F=f)
                        log_artifact(path_png)
                        log_artifact(path_pdf)

                        test_names = folds_test[f]
                        print(f"\nEvaluation ... [{feature_type.upper()}]")
                        model.eval()
                        with torch.no_grad():

                            y_groundTruth = []
                            score = []
                            for name in test_names:
                                feat = making_sequence(np.asarray(hf[name]), timestep=TIMESTEP)
                                feat = torch.from_numpy(np.transpose(feat, axes=(0, 2, 1))).type(torch.FloatTensor).to(device)
                                take = label[:,0] == name

                                y_tmp = label[:,-1].astype(float)[take]
                                if len(y_tmp) > 1: # Verifying duplicated samples
                                    raise ValueError(f"Duplicated test sample (name: {name})")
                                y_groundTruth.append(y_tmp[0])
                                output = torch.sigmoid(model(feat)).mean().item()
                                score.append(output)

                            print(f"\nFold {f+1} Run {r}")
                            metric_fold = computing_performance(score,metric_fold, label=y_groundTruth, K=f,R = r)

                log_metric("accuracy_average", round(np.mean(metric_fold["accuracy"]),2))
                log_metric("accuracy_std", round(np.std(metric_fold["accuracy"]),2))

                log_metric("recall_average", round(np.mean(metric_fold["recall"]),2))
                log_metric("recall_std", round(np.std(metric_fold["recall"]),2))

                log_metric("precision_average", round(np.mean(metric_fold["precision"]),2))
                log_metric("precision_std", round(np.std(metric_fold["precision"]),2))

                log_metric("f1_average", round(np.mean(metric_fold["f1"]),2))
                log_metric("f1_std", round(np.std(metric_fold["f1"]),2))

                log_metric("auc_average", round(np.mean(metric_fold["auc"]),2))
                log_metric("auc_std", round(np.std(metric_fold["auc"]),2))

                print("Saving heapmat of accuracy values through folds and runs ...")
                figure_path_pdf,  figure_path_png = heat_map(metric_fold["accuracy"])

                log_artifact(figure_path_pdf)
                log_artifact(figure_path_png)

