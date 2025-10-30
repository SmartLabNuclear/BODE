# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 23:42:10 2025

@author: zaidabulawi
"""

import os
import sys
import json
import random
import time
import numpy as np
from glob import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.ndimage
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.DenseNet_ensemble import DenseNet
from utils.DataLoader_2D import StarDataset
from save_stats import save_stats

# Device setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")
if 'cuda' in device.type:
    print(f" CUDA Device Name: {torch.cuda.get_device_name(device)}")

# Loss and activation
nll_loss = lambda mu, sigma, y: torch.log(sigma)/2 + ((y - mu)**2) / (2 * sigma)
sp = torch.nn.Softplus()

# Training configuration
epochs = 200

# Define hyperparameter search space
hyperparam_space = {
    "lr": lambda: 10 ** random.uniform(-4, -2),
    "weight_decay": lambda: 10 ** random.uniform(-4, -2),
    "drop_rate": lambda: random.uniform(0, 0.5),
    "growth_rate": lambda: random.choice([8, 12, 16, 24, 32, 48]),
    "batch_size": lambda: random.choice([8, 12, 16, 24, 32, 48, 64, 128]),
    "init_features": lambda: random.choice([8, 12, 16, 24, 32, 48, 64, 128]),
    "num_blocks": lambda: random.choice([3, 5]),
    "blocks_1_layers": lambda: random.choice(range(3, 10)),
    "blocks_2_layers": lambda: random.choice(range(3, 10)),
    "blocks_3_layers": lambda: random.choice(range(3, 10)),
    "blocks_4_layers": lambda: random.choice(range(3, 10)),
    "blocks_5_layers": lambda: random.choice(range(3, 10)),
}

# Model evaluation function
def test_bayesian(hyperparameters, processor_id):
    # Set unique seed
    seed = processor_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_start_time = time.time()

    # Extract hyperparams
    lr = hyperparameters["lr"]
    weight_decay = hyperparameters["weight_decay"]
    drop_rate = hyperparameters["drop_rate"]
    growth_rate = hyperparameters["growth_rate"]
    batch_size = hyperparameters["batch_size"]
    init_features = hyperparameters["init_features"]
    num_blocks = hyperparameters['num_blocks']
    blocks = [hyperparameters[f'blocks_{i}_layers'] for i in range(1, num_blocks + 1)]

    print(f"\n Training model {processor_id} with hyperparameters:\n{json.dumps(hyperparameters, indent=2)}")

    # Load data
    input_filelist = sorted(glob('./sample/input_*.pt'))
    output_filelist = sorted(glob('./sample/output_*.pt'))
    random.seed(1234)
    num_files_to_select = int(0.3 * len(input_filelist))
    random_indices = random.sample(range(len(input_filelist)), num_files_to_select)
    selected_input_filelist = [input_filelist[i] for i in sorted(random_indices)]
    selected_output_filelist = [output_filelist[i] for i in sorted(random_indices)]
    full_index = range(len(selected_input_filelist))
    train_index = random.sample(range(len(selected_input_filelist)), int(0.7 * len(selected_input_filelist)))
    test_index = list(set(full_index) - set(train_index))
    input_filelist_train = [selected_input_filelist[i] for i in sorted(train_index)]
    output_filelist_train = [selected_output_filelist[i] for i in sorted(train_index)]
    input_filelist_test = [selected_input_filelist[i] for i in sorted(test_index)]
    output_filelist_test = [selected_output_filelist[i] for i in sorted(test_index)]
    save_stats(input_filelist_train, output_filelist_train, output_filelist_test)

    input_train_mean = torch.load('input_train_mean.pt')
    input_train_std = torch.load('input_train_std.pt')
    output_train_var = torch.load('output_train_var.pt')
    output_test_var = torch.load('output_test_var.pt')

    train_dataset = StarDataset(input_filelist_train, output_filelist_train)
    test_dataset = StarDataset(input_filelist_test, output_filelist_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = DenseNet(in_channels=4, out_channels=2, blocks=blocks, growth_rate=growth_rate,
                     init_features=init_features, drop_rate=drop_rate, bn_size=4, bottleneck=True,
                     out_activation='Softplus').to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-6)

    for epoch in range(epochs):
        model.train()
        nll_total, mse = 0.0, 0.0
        total_samples = 0
        for input_features, output_qois in train_loader:
            input_features = (input_features - input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            input_features, output_qois = input_features.to(device), output_qois.to(device)
            batch_size_actual = output_qois.size(0)
            total_samples += batch_size_actual

            noise = np.random.normal(0, 0.1, size=output_qois.shape)
            noise = scipy.ndimage.gaussian_filter(noise, sigma=12)
            rescaled = torch.from_numpy((0.1 / np.std(noise)) * noise).to(output_qois.device).to(output_qois.dtype)
            output_qois_noisy = torch.clamp(rescaled * output_qois + output_qois, min=0)

            optimizer.zero_grad()
            output = model(input_features)
            mu, sig = output[:,0,:,:], sp(output[:,1,:,:]) + 1e-6
            loss = torch.sum(nll_loss(mu, sig, output_qois_noisy.squeeze(1)))
            loss.backward()
            optimizer.step()
            nll_total += loss.item()
            mse += F.mse_loss(mu, output_qois_noisy.squeeze(1), reduction='sum').item()

        scheduler.step(nll_total / len(train_loader))
        avg_rmse = np.sqrt(mse / (total_samples * output_qois[0].numel()))
        print(f" Epoch {epoch+1}/{epochs}: NLL = {nll_total:.4f}, MSE = {mse:.4f}, Avg RMSE = {avg_rmse:.6f}")
    with torch.no_grad():
        model.eval()
        nll_total, mse = 0.0, 0.0
        for input_features, output_qois in test_loader:
            input_features = (input_features - input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            input_features, output_qois = input_features.to(device), output_qois.to(device)
            noise = np.random.normal(0, 0.1, size=output_qois.shape)
            noise = scipy.ndimage.gaussian_filter(noise, sigma=12)
            rescaled = torch.from_numpy((0.1 / np.std(noise)) * noise).to(output_qois.device).to(output_qois.dtype)
            output_qois_noisy = torch.clamp(rescaled * output_qois + output_qois, min=0)

            output = model(input_features)
            mu, sig = output[:,0,:,:], sp(output[:,1,:,:]) + 1e-6
            loss = torch.sum(nll_loss(mu, sig, output_qois_noisy.squeeze(1)))
            nll_total += loss.item()
            mse += F.mse_loss(mu, output_qois_noisy.squeeze(1), reduction='sum').item()

        rmse_test = np.sqrt(mse / len(test_index) / test_loader.dataset[0][1].numel())
        model_duration = time.time() - model_start_time
        print(f" Model {processor_id} complete. RMSE = {rmse_test:.4f} (Time: {model_duration:.2f}s)")
        return rmse_test, seed


# ===========================
# Main Script
# ===========================
start_time = time.time()

N_RANDOM_SEARCH = 100
K_GREEDY_SELECT = 20

all_candidates = []
for i in range(N_RANDOM_SEARCH):
    random.seed(i+1)
    params = {key: sampler() for key, sampler in hyperparam_space.items()}
    score, used_seed = test_bayesian(params, i+1)
    all_candidates.append({"params": params, "score": score, "seed": used_seed})

sorted_candidates = sorted(all_candidates, key=lambda x: x["score"])
selected_ensemble = sorted_candidates[:K_GREEDY_SELECT]

with open("hyperdeep_ensemble_config.json", "w") as f:
    json.dump(selected_ensemble, f, indent=2)

print("\n Final Selected Ensemble:")
for i, model in enumerate(selected_ensemble):
    print(f"Model {i+1}: score = {model['score']:.4f}, seed = {model['seed']}, params = {model['params']}")

total_duration = time.time() - start_time
print(f"\n Total script time: {total_duration:.2f} seconds.")
