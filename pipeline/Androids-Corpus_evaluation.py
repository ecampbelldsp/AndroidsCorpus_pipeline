#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  1/9/23 8:47

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import os
os.chdir("../")

import datetime
import numpy as np
import random
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from sklearn.metrics import top_k_accuracy_score, accuracy_score
from sklearn import metrics
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from mlflow import log_metric,log_metrics, log_param, log_params, log_artifacts
import mlflow

from config.config import FEATURE_PARMS
from ai import DeepAudioNet


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

def computing_performance(score, prediction_task="binary",label = None, K = None, params = None, result_folder_runs = None):
        metric_fold = {"accuracy": None, "recall": None, "precision": None, "auc": None, "f1": None, "folds_name": None}
        # with open(f"{result_folder_runs}{K}_performance.txt", "w") as file:
        y_truth = np.asarray(label, dtype=float).astype(int)
        fpr, tpr, thresholds = metrics.roc_curve(y_truth, score, pos_label=1)

        # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        estimation_final = np.zeros_like(score, dtype=int)
        pos = np.where(score >= optimal_threshold)
        estimation_final[pos] = 1

        # estimation_final, optimal_threshold, accuracy_max = final_estimation(score, y_truth)
        metric_fold["folds_name"] =K
        acc_final = round(metrics.accuracy_score(y_truth, estimation_final) * 100, 2)
        metric_fold["accuracy"] = acc_final

        precision_final = round(metrics.precision_score(y_truth, estimation_final, average = "macro") * 100, 2)
        metric_fold["precision"] = precision_final

        f1_final = round(metrics.f1_score(y_truth, estimation_final, average = "macro") * 100, 2)
        metric_fold["f1"] = f1_final

        recall_final = round(metrics.recall_score(y_truth, estimation_final, average = "macro") * 100, 2)
        metric_fold["recall"] = recall_final

        auc_final = round(metrics.auc(fpr, tpr) * 100, 2)
        metric_fold["auc"] = auc_final

        print(metrics.classification_report(y_truth, estimation_final, target_names=['0', '1'], digits=4))

        # file.write(metrics.classification_report(y_truth, estimation_final, target_names=['0', '1'], digits=4))
        # file.write("\n\n\n")

        print("Accuracy: " + str(acc_final))
        # file.write("Accuracy: " + str(acc_final) + "\n")

        print("AUC: " + str(auc_final))
        # file.write("AUC: " + str(auc_final))



        linea = np.linspace(0, 1, len(fpr))

        # plt.figure()
        # plt.plot(fpr, tpr, 'r')
        # plt.plot(linea, linea, 'b--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('AUC = ' + str(auc_final))
        # plt.savefig(result_folder_runs + str(K) + "_AUC.png")
        # plt.close()


        # file.write("\n\n\n")
        # file.write("\n\n Training variables   Fold-" + str(K) + "\n\n")
        # file.write(str(params))

        return metric_fold

if __name__ == "__main__":

    RUNS = 10
    TASK_LIST = ["Interview-Task"]  # , "Reading-Task"
    CORPUS = "Androids-Corpus"
    feature_type = "melSpectrum"


    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    for task in TASK_LIST:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = f"{CORPUS}-{task}"
        # Define hyperparameters
        metric_fold_collection = {"accuracy": [], "recall": [], "precision": [], "auc": [], "f1": []}
        batch_size = 128
        learning_rate = 0.001
        num_epochs = 1
        validation_rate = 0.3

        params = {"batch_size": batch_size, "learning_rate": learning_rate, "Epochs": num_epochs,
                  "validation_rate": validation_rate, "runs":RUNS}


        TIMESTEP = FEATURE_PARMS[feature_type][1]

        metric_fold = []
        for t in range(RUNS):
            ID = f"{datetime.datetime.now()}"
            # mlflow.end_run()
            with mlflow.start_run(run_name=f"{ID}_{t}") as run:
                log_params(params)
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
                            feat = making_sequence(np.asarray(hf[name]), timestep = TIMESTEP)#[:20,:,:]
                            x_train.append(feat)
                            y_train.append([label[j,-1].astype(float)]*feat.shape[0])
                x_train = np.concatenate(x_train)
                x_train = np.transpose(x_train,axes=(0, 2, 1))

                y_train = np.concatenate(y_train)[:,np.newaxis]

                print("Data loaded on CPU")
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
                    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")

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

                #TODO Load Test set
                print("Evaluating model performance")
                for f,test_names in enumerate(folds_test):

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
                    #TODO Evaluating Test set
                    print(f"Fold {f}")
                    metric_fold = computing_performance(score, label=y_groundTruth, K=f)
                    log_metrics(metric_fold,step=f)

                    metric_fold_collection["accuracy"].append(metric_fold["accuracy"])
                    metric_fold_collection["recall"].append(metric_fold["recall"])
                    metric_fold_collection["precision"].append(metric_fold["precision"])
                    metric_fold_collection["f1"].append(metric_fold["f1"])
                    metric_fold_collection["auc"].append(metric_fold["auc"])

                    # Log an artifact (output file)
                    if not os.path.exists("outputs"):
                        os.makedirs("outputs")
                    with open("outputs/test.txt", "w") as f:
                        f.write("hello world!")
                    log_artifacts("outputs")

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


