#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  1/9/23 8:47

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import gc
import os
os.chdir("../")

import datetime
import numpy as np
import random
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from mlflow import log_metric,log_metrics, log_param, log_params, log_artifacts
import mlflow

from config.config import FEATURE_PARMS
from ai import DeepAudioNet
from evaluation.performance import computing_performance

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
if device == "cuda":
    try:
        torch.cuda.set_device(0)
        print("GPU " + str(0))
    except IndexError:
        torch.cuda.set_device(0)
        print("GPU 0")

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


if __name__ == "__main__":

    RUNS = 10
    TASK_LIST = ["Interview-Task"]  # , "Reading-Task"
    CORPUS = "Androids-Corpus"
    feature_type = "melSpectrum"#"melSpectrum" compare_lld

    if not os.path.exists("model"):
        os.mkdir("model")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    for task in TASK_LIST:
        #Setting Mlflow experiment name
        os.environ["MLFLOW_EXPERIMENT_NAME"] = f"{CORPUS}-{task}"
        # Define hyperparameters
        metric_fold_collection = {"accuracy": [], "recall": [], "precision": [], "auc": [], "f1": []}
        batch_size = 128
        learning_rate = 0.001
        num_epochs = 100
        validation_rate = 0.3
        EARLY_STOP = {"Do":True, "max_attempts": 10}
        TIMESTEP = FEATURE_PARMS[feature_type][1]

        params = {"batch_size": batch_size, "learning_rate": learning_rate, "Epochs": num_epochs,"Early_stop": EARLY_STOP,
                  "validation_rate": validation_rate, "runs":RUNS,"Feature": feature_type,"Sequence length":TIMESTEP}

        metric_fold = []
        for t in range(RUNS):
            ID = f"{datetime.datetime.now()}"
            # mlflow.end_run()
            with mlflow.start_run(run_name=f"{ID}_{t}") as run:
                log_params(params)
                label = np.genfromtxt(f"label/label_{CORPUS}_{task}.txt", dtype=str, delimiter=" ")[1:,:]
                # test_name =set(list(label[:,0]))
                folds = np.genfromtxt(f"default-folds_Androids-Corpus/fold_{task}.txt", delimiter=",",dtype=str)
                hf = h5py.File(f"features/{feature_type}_{CORPUS}_{task}.h5", 'r')

                #Creating K-Folds and loading features
                folds_test = []
                folds_train = [[] for i in range(folds.shape[0])]
                for i in range(folds.shape[0]):
                    x_train = []
                    y_train = []
                    folds_test.append(folds[i].split(" "))
                    for j, name in enumerate(label[:,0]):
                        if not name in folds[i]:
                            folds_train[i].append(name)
                            feat = making_sequence(np.asarray(hf[name]), timestep = TIMESTEP)#[:20,:,:]
                            x_train.append(feat)
                            y_train.append([label[j,-1].astype(float)]*feat.shape[0])
                    x_train = np.concatenate(x_train)
                    x_train = np.transpose(x_train,axes=(0, 2, 1))

                    y_train = np.concatenate(y_train)[:,np.newaxis]

                    print("Data loaded")
                    index = list(range(x_train.shape[0]))
                    random.shuffle(index)

                    val_index = index[ : int(len(index) * validation_rate) ]
                    index_train = index[int(len(index) * validation_rate) : ]

                    x_val = x_train[val_index,:,:]
                    y_val = y_train[val_index]
                    x_train = x_train[index_train,:,:]
                    y_train = y_train[index_train]

                    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
                    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)

                    x_val = torch.from_numpy(x_val).type(torch.FloatTensor)
                    y_val = torch.from_numpy(y_val).type(torch.FloatTensor)

                    # Create DataLoader instances for training and validation
                    train_data = TensorDataset(x_train, y_train)
                    val_data = TensorDataset(x_val,y_val)
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

                    # TODO Initialize model


                    model = DeepAudioNet.CustomMel1(in_channels=x_train.shape[1], outputs=1).to(device)
                    # Define loss function and optimizer
                    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for binary classification
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    # TODO Training loop

                    # Training loop
                    MIN_VALIDATION_LOSS = np.inf
                    if EARLY_STOP["Do"]:
                        consecutive_attempt = 0
                    for epoch in range(num_epochs):
                        model.train()
                        total_loss = 0.0

                        for inputs, targets in train_loader:
                            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                        avg_loss = total_loss / len(train_loader)
                        print(f"Run {t+1} Fold {i+1} Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")
                        # Validation loop
                        model.eval()
                        total_val_loss = 0
                        total_correct = 0
                        total_samples = 0

                        with torch.no_grad():
                            for inputs, targets in val_loader:
                                inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
                                outputs = model(inputs)
                                predictions = torch.round(torch.sigmoid(outputs))
                                total_correct += (predictions == targets).sum().item()
                                total_samples += targets.size(0)

                                loss = criterion(outputs, targets)
                                total_val_loss += loss.item()

                        validation_accuracy = total_correct / total_samples
                        avg_loss = total_val_loss / len(val_loader)
                        print(f"Validation Accuracy: {validation_accuracy * 100:.2f}% | Validation Loss: {avg_loss:.4f}")
                        print("--" * 10)
                        if avg_loss < MIN_VALIDATION_LOSS:
                            print(f"Validation loss improved from {MIN_VALIDATION_LOSS:.4f} to {avg_loss:.4f}")
                            print(":)"*10)
                            MIN_VALIDATION_LOSS = avg_loss
                            consecutive_attempt = 0
                            torch.save(model.state_dict(), "model/tmp.pth")
                        else:
                            consecutive_attempt += 1
                        if EARLY_STOP["Do"] and consecutive_attempt > EARLY_STOP["max_attempts"]:
                            print("Early stop")
                            break
                    #Loading best model
                    model.load_state_dict(torch.load(f"model/tmp.pth"))

                    test_names = folds_test[i]
                    print("\nEvaluating model performance")
                    # for f,test_names in enumerate(folds_test):
                    with torch.no_grad():
                        model.eval()
                        # Freeing GPU memory
                        del inputs, targets
                        gc.collect()
                        torch.cuda.empty_cache()

                        y_groundTruth = []
                        score = []
                        for name in test_names:
                            feat = making_sequence(np.asarray(hf[name]), timestep=TIMESTEP)
                            feat = torch.from_numpy(np.transpose(feat, axes=(0, 2, 1))).type(torch.FloatTensor).to(device)
                            take = label[:,0] == name
                            y_tmp = label[:,-1].astype(float)[take][0]
                            y_groundTruth.append(y_tmp)
                            output = torch.sigmoid(model(feat)).mean().item()
                            score.append(output)

                        print(f"\nFold {i+1} Run {t}")
                        metric_fold = computing_performance(score, label=y_groundTruth, K=i)
                        log_metrics(metric_fold,step=i)

                        metric_fold_collection["accuracy"].append(metric_fold["accuracy"])
                        metric_fold_collection["recall"].append(metric_fold["recall"])
                        metric_fold_collection["precision"].append(metric_fold["precision"])
                        metric_fold_collection["f1"].append(metric_fold["f1"])
                        metric_fold_collection["auc"].append(metric_fold["auc"])



        with mlflow.start_run(run_name=f"{ID}_{RUNS}") as run:
            log_params(params)

            log_metric("accuracy_average", np.mean(metric_fold_collection["accuracy"]))
            log_metric("accuracy_std", np.std(metric_fold_collection["accuracy"]))

            log_metric("recall_average", np.mean(metric_fold_collection["recall"]))
            log_metric("recall_std", np.std(metric_fold_collection["recall"]))

            log_metric("precision_average", np.mean(metric_fold_collection["precision"]))
            log_metric("precision_std", np.std(metric_fold_collection["precision"]))


            log_metric("f1_average", np.mean(metric_fold_collection["f1"]))
            log_metric("f1_std", np.std(metric_fold_collection["f1"]))


            log_metric("auc_average", np.mean(metric_fold_collection["auc"]))
            log_metric("auc_std", np.std(metric_fold_collection["auc"]))


