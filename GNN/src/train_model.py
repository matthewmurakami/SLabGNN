import os
import torch
import torch_geometric
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss
from tqdm import tqdm
import math
import matplotlib.pyplot as plt


def train(model, train_loader, optimizer, criterion):
    model.train()
    for data in train_loader:
        data = data.to(model.device)  # Move data to the same device as the model
        out = model(data)
        out = out.view(-1)  # Flatten the output to match target shape
        if out.size() != data.y.size():
            print(f"Output shape: {out.size()}, Target shape: {data.y.size()}")
            exit()
        loss = criterion(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, loader, criterion, print_met=True):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(model.device)
            out = model(data)
            out = out.view(-1) 
            if out.size() != data.y.size():
                print(f"Output shape: {out.size()}, Target shape: {data.y.size()}")
                exit()
            loss = criterion(out, data.y)
            total_loss += loss.item()

            if print_met:
                print(f"Predicted: {out}, True: {data.y}, RMSE: {math.sqrt(loss.item())}")

    avg_loss = total_loss / len(loader.dataset)
    return math.sqrt(avg_loss)

def train_model(train_loader, val_loader, model, output_filepath, learning_rate, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.device = device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs):
        train(model, train_loader, optimizer, criterion)
        train_rmse = test(model, train_loader, criterion, False)
        val_rmse = test(model, val_loader, criterion, False)

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)

        print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')
        print()
    
    torch.save(model.state_dict(), output_filepath)
    print("saved the model to: ", output_filepath)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training RMSE')
    plt.plot(val_losses, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.show()

"""
def train(model, train_loader, optimizer, criterion):
    model.train()
    for data in train_loader:
        out = model(data)
        loss = criterion(out[0], data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, loader, criterion, print_met=True):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = criterion(out[0], data.y)
            total_loss += loss.sum().item()  # Sum loss over the batch

            if print_met:
                print(f"Predicted: {out}, True: {data.y}, RMSE: {math.sqrt(loss)}")

    avg_loss = total_loss / len(loader.dataset)  # average loss for batch
    return math.sqrt(avg_loss)  # Return RMSE

def train_model(train_loader, val_loader, model, output_filepath, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, 1000):
        train(model, train_loader, optimizer, criterion)
        train_rmse = test(model, train_loader, criterion, False)
        val_rmse = test(model, val_loader, criterion, False)

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)

        print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')
        print()

    # Plot the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training RMSE')
    plt.plot(val_losses, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.show()
"""