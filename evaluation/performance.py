#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  5/9/23 2:06

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""

import numpy as np
from sklearn import metrics
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
