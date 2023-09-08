#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  5/9/23 22:17

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import gc
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

def train_CNN_LSTM(model, params, train=None, val=None, r = None, f = None):
    def print_progress_bar(iteration, total, bar_length=50):
        progress = (iteration / total)
        arrow = '=' * int(round(bar_length * progress))
        spaces = ' ' * (bar_length - len(arrow))
        print(f'\r[{arrow + spaces}] {int(progress * 100)}%', end=" ")

    x_train, y_train = train[0], train[1]
    x_val, y_val = val[0], val[1]

    feature_type = params["Feature"]
    batch_size = params["batch_size"]
    EARLY_STOP = params["Early_stop"]
    learning_rate = params["learning_rate"]
    num_epochs = params["Epochs"]
    device = params["device"]
    # Create DataLoader instances for training and validation
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Training ...\n")
    # Training loop
    MIN_VALIDATION_LOSS = np.inf
    if EARLY_STOP["Do"]:
        consecutive_attempt = 0
    loss_train_plot = []
    loss_val_plot = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for i, (inputs, targets) in enumerate(train_loader):
            # Update the progress bar
            print_progress_bar(i + 1, len(train_loader))

            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_train_plot.append(avg_loss)

        print(f"\n{feature_type.upper()} Run {r + 1} Fold {f + 1}")
        print(f"Training Loss: {avg_loss:.4f}")
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
        loss_val_plot.append(avg_loss)

        print(f"Validation Loss: {avg_loss:.4f} ( Accuracy: {validation_accuracy * 100:.2f}% )")
        if avg_loss < MIN_VALIDATION_LOSS:
            print("--" * 30)
            print(f"[ Validation loss improved from {MIN_VALIDATION_LOSS:.4f} to {avg_loss:.4f} ]")
            print("--" * 30)
            MIN_VALIDATION_LOSS = avg_loss
            consecutive_attempt = 0
            torch.save(model.state_dict(), "model/tmp.pth")
        else:
            consecutive_attempt += 1
        if EARLY_STOP["Do"] and consecutive_attempt > EARLY_STOP["max_attempts"]:
            print("Early stop activated\n")
            break
    # Loading best model and deleting temporal file

    print("Loading best model ...")
    model.load_state_dict(torch.load(f"model/tmp.pth"))
    os.remove("model/tmp.pth")
    with torch.no_grad():
        # Freeing GPU memory
        del inputs, targets
        gc.collect()
        torch.cuda.empty_cache()
    return model, [loss_train_plot, loss_val_plot]