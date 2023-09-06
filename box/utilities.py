#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  5/9/23 2:06

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""

import numpy as np
import random
from sklearn import metrics
import torch

def select_gpu_with_most_free_memory():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPU amount: {num_gpus}")
        if num_gpus >= 1:
            gpu_memory = []

            for i in range(num_gpus):
                gpu = torch.cuda.get_device_properties(i)
                # free_memory = gpu.total_memory - gpu.memory_allocated()
                free_memory = torch.cuda.mem_get_info(device = i)[0]
                gpu_memory.append((i, free_memory))

            gpu_memory.sort(key=lambda x: x[1], reverse=True)
            selected_gpu = gpu_memory[0][0]  # Choose the GPU with the most free memory

            print(f"Selected GPU: {selected_gpu}")
            torch.cuda.set_device(selected_gpu)  # Set the selected GPU as the active device
        else:
            print("Only one GPU available. Using GPU 0.")
            torch.cuda.set_device(0)  # Use the only available GPU
            selected_gpu = 0
        return selected_gpu,"cuda"
    else:
        print("No GPU available. Using CPU.")
        return None,"cpu"

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

def split_train_data(x_train,y_train,validation_rate):
    x_train = np.concatenate(x_train)
    x_train = np.transpose(x_train, axes=(0, 2, 1))
    y_train = np.concatenate(y_train)[:, np.newaxis]

    index = list(range(x_train.shape[0]))
    random.shuffle(index)

    index_val = index[: int(len(index) * validation_rate)]
    index_train = index[int(len(index) * validation_rate):]

    x_val = x_train[index_val, :, :]
    y_val = y_train[index_val]
    x_train = x_train[index_train, :, :]
    y_train = y_train[index_train]

    return x_train,y_train,x_val,y_val

def computing_performance(score, prediction_task="binary", label=None, K=None, params=None, result_folder_runs=None):
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
    metric_fold["folds_name"] = K
    acc_final = round(metrics.accuracy_score(y_truth, estimation_final) * 100, 2)
    metric_fold["accuracy"] = acc_final

    precision_final = round(metrics.precision_score(y_truth, estimation_final, average="macro") * 100, 2)
    metric_fold["precision"] = precision_final

    f1_final = round(metrics.f1_score(y_truth, estimation_final, average="macro") * 100, 2)
    metric_fold["f1"] = f1_final

    recall_final = round(metrics.recall_score(y_truth, estimation_final, average="macro") * 100, 2)
    metric_fold["recall"] = recall_final

    auc_final = round(metrics.auc(fpr, tpr) * 100, 2)
    metric_fold["auc"] = auc_final

    print(metrics.classification_report(y_truth, estimation_final, target_names=['0', '1'], digits=4))

    print("Accuracy: " + str(acc_final))

    print("AUC: " + str(auc_final))

    return metric_fold
