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
from box.utilities import computing_performance, making_sequence, split_train_data, select_gpu_with_most_free_memory

device = "cuda" if torch.cuda.is_available() else "cpu"
selected_GPU, device = select_gpu_with_most_free_memory()


if __name__ == "__main__":

    RUNS = 10
    TASK_LIST = ["Interview-Task"]  # , "Reading-Task"
    CORPUS = "Androids-Corpus"
    FEATURE_TYPE_LIST = ["melSpectrum", "rasta", "compare_lld","egemap_lld", "hubert_base", "wav2vec2_base"]
    # feature_type = "melSpectrum"#"melSpectrum" compare_lld egemap_lld rasta hubert_base wav2vec2_base
    for feature_type in FEATURE_TYPE_LIST:
        if not os.path.exists("model"):
            os.mkdir("model")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        for task in TASK_LIST:
            #Setting Mlflow experiment name
            os.environ["MLFLOW_EXPERIMENT_NAME"] = f"{CORPUS}-{task} - {feature_type}"
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
            for r in range(RUNS):
                ID = f"{datetime.datetime.now()}"
                with mlflow.start_run(run_name=f"{ID}_{r}") as run:
                    log_params(params)
                    label = np.genfromtxt(f"label/label_{CORPUS}_{task}.txt", dtype=str, delimiter=" ")[1:,:]

                    #Loading default distribution folds files
                    folds = np.genfromtxt(f"default-folds_Androids-Corpus/fold_{task}.txt", delimiter=",",dtype=str)
                    hf = h5py.File(f"features/{feature_type}_{CORPUS}_{task}.h5", 'r')

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

                        # Create DataLoader instances for training and validation
                        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(TensorDataset(x_val,y_val), batch_size=batch_size, shuffle=False)

                        print("Model initialization")
                        model = DeepAudioNet.CustomMel1(in_channels=x_train.shape[1], outputs=1).to(device)
                        # Define loss function and optimizer
                        criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for binary classification
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                        print("Training ...\n")
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
                            print(f"Run {r+1} Fold {f+1} Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")
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
                                print(f"[  Validation loss improved from {MIN_VALIDATION_LOSS:.4f} to {avg_loss:.4f}  ]\n")
                                # print(":)"*10)
                                MIN_VALIDATION_LOSS = avg_loss
                                consecutive_attempt = 0
                                torch.save(model.state_dict(), "model/tmp.pth")
                            else:
                                consecutive_attempt += 1
                            if EARLY_STOP["Do"] and consecutive_attempt > EARLY_STOP["max_attempts"]:
                                print("Early stop activated\n")
                                break

                        print("Loading best model ...")
                        #Loading best model and deleting temporal file
                        model.load_state_dict(torch.load(f"model/tmp.pth"))
                        os.remove("model/tmp.pth")

                        test_names = folds_test[f]
                        print("\nEvaluation ...")
                        model.eval()
                        with torch.no_grad():
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

                                y_tmp = label[:,-1].astype(float)[take]
                                if len(y_tmp) > 1: # Verifying duplicated samples
                                    raise ValueError(f"Duplicated test sample (name: {name})")
                                y_groundTruth.append(y_tmp[0])
                                output = torch.sigmoid(model(feat)).mean().item()
                                score.append(output)

                            print(f"\nFold {f+1} Run {r}")
                            metric_fold = computing_performance(score, label=y_groundTruth, K=f)
                            log_metrics(metric_fold,step=f)

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


