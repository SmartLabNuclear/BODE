# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 23:43:15 2025
@author: zaidabulawi
"""

import os
import sys
import json
import random
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
from args import args
import time

# Use cuda:1 as specified
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loss and activation
nll_loss = lambda mu, sigma, y: torch.log(sigma)/2 + ((y-mu)**2)/(2*sigma)
sp = torch.nn.Softplus()

# Training configuration
epochs = 200
POP_SIZE = 20
GENERATIONS = 7
MUTATION_RATE = 0.4

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

def test_bayesian(hyperparameters, processor_id, generation=0, model_index=0, seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    lr = hyperparameters["lr"]
    weight_decay = hyperparameters["weight_decay"]
    drop_rate = hyperparameters["drop_rate"]
    growth_rate = hyperparameters["growth_rate"]
    batch_size = hyperparameters["batch_size"]
    init_features = hyperparameters["init_features"]
    num_blocks = hyperparameters['num_blocks']
    blocks = [hyperparameters[f'blocks_{i}_layers'] for i in range(1, num_blocks + 1)]

    print(f"\n>>> Starting Gen {generation}, Model {model_index} <<<")
    print(f"Seed: {seed}")
    print("Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")

    input_filelist = sorted(glob('./sample/input_*.pt'))
    output_filelist = sorted(glob('./sample/output_*.pt'))
    random.seed(seed)
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,
                                  threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-6)

    for epoch in range(epochs):
        model.train()
        nll_total, mse, rmse_total = 0.0, 0.0, 0.0
        batch_count = 0
        for input_features, output_qois in train_loader:
            input_features = (input_features - input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            input_features, output_qois = input_features.to(device), output_qois.to(device)
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
            mse_batch = F.mse_loss(mu, output_qois_noisy.squeeze(1), reduction='sum').item()
            mse += mse_batch
            rmse_total += np.sqrt(mse_batch / output_qois[0].numel())
            batch_count += 1

        scheduler.step(nll_total / len(train_loader))
        print(f"Gen {generation}, Model {model_index}, Epoch {epoch+1}/{epochs} – NLL: {nll_total / batch_count:.4f}, MSE: {mse / batch_count:.4f}, RMSE: {rmse_total / batch_count:.4f}")

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
        print(f">>> Gen {generation}, Model {model_index} – Final RMSE: {rmse_test:.4f}")
        return rmse_test


start_time = time.time()
population = []
for i in range(POP_SIZE):
    seed = i + 1
    params = {k: v() for k, v in hyperparam_space.items()}
    score = test_bayesian(params, processor_id=i+1, generation=0, model_index=i+1, seed=seed)
    population.append({"params": params, "score": score, "seed": seed})

for gen in range(GENERATIONS):
    print("="*60)
    print(f"Evolution Generation {gen+1}/{GENERATIONS}")
    print("="*60)
    population = sorted(population, key=lambda x: x['score'])[:POP_SIZE]
    next_gen = population[:POP_SIZE//2]  # Keep parents with their seeds
    model_counter = len(next_gen)
    next_seed = POP_SIZE * (gen + 1) + 1  # Seeds for new children

    while len(next_gen) < POP_SIZE:
        parent_model = random.choice(next_gen)
        parent = parent_model["params"]
        child = parent.copy()
        for key in child:
            if random.random() < MUTATION_RATE:
                child[key] = hyperparam_space[key]()
        model_counter += 1
        seed = next_seed
        next_seed += 1
        score = test_bayesian(child, processor_id=random.randint(1, 999), generation=gen+1, model_index=model_counter, seed=seed)
        next_gen.append({"params": child, "score": score, "seed": seed})

    population = next_gen

population = sorted(population, key=lambda x: x['score'])
final_ensemble = population[:20]

with open("evolved_ensemble_config.json", "w") as f:
    json.dump(final_ensemble, f, indent=2)

print("\nFinal Evolved Ensemble:")
for i, model in enumerate(final_ensemble):
    print(f"Model {i+1}: score = {model['score']:.4f}, seed = {model['seed']}, params = {model['params']}")

total_duration = time.time() - start_time
print(f"\n Total script time: {total_duration:.2f} seconds.")
