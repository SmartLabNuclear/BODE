# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:03:26 2024

@author: zaidabulawi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:49:17 2024

@author: zaidabulawi
"""

import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.DenseNet_ensemble import DenseNet
from utils.DataLoader_2D import StarDataset
# from utils.plot import plot_prediction_det, save_stats
import json
from time import time
import random
from args import args, device
import pickle
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch._tensor import Tensor
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.utils.tutorials.cnn_utils import evaluate, load_mnist, train
from ax.plot.contour import plot_contour
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import time
from save_stats import save_stats

logger = {}
logger['nll_train'] = []
logger['nll_test'] = []
logger['nll_adv'] = []
logger['r2_train'] = []
logger['r2_test'] = []
logger['r2_adv'] = []
logger['rmse_train'] = []
logger['rmse_test'] = []
logger['rmse_adv'] = []

nll_loss = lambda mu, sigma, y: torch.log(sigma)/2 + ((y-mu)**2)/(2*sigma)
sp = torch.nn.Softplus()

epochs = 200

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_bayesian(hyperparameters, processor_id):
    
    #lr, growth_rate = hyperparameters
    lr = hyperparameters["lr"]
    weight_decay = hyperparameters["weight_decay"]
    drop_rate = hyperparameters["drop_rate"]
    growth_rate = hyperparameters["growth_rate"]
    batch_size = hyperparameters["batch_size"]
    init_features = hyperparameters["init_features"]
    num_blocks = hyperparameters['num_blocks']
    blocks = [hyperparameters[f'blocks_{i}_layers'] for i in range(1, num_blocks + 1)]
    print("lr, weight decay, drop rate, growth rate, batch size, initial features, epochs, and blocks are: ")
    print(lr, " ", weight_decay, " ", drop_rate, " ", growth_rate, " ", batch_size, " ", init_features, " ", epochs, blocks)
    

    # load data
    input_filelist = sorted(glob('./sample/input_*.pt'))
    output_filelist = sorted(glob('./sample/output_*.pt'))

    random.seed(processor_id + 1)
    " select data files for training, randomly "

    num_files_to_select = int(0.3 * len(input_filelist))
    random_indices = random.sample(range(len(input_filelist)), num_files_to_select)
    selected_input_filelist = [input_filelist[i] for i in sorted(random_indices)]
    selected_output_filelist = [output_filelist[i] for i in sorted(random_indices)]

    full_index = range(len(selected_input_filelist))

    train_index = random.sample(range(len(selected_input_filelist)),int(0.7*len(selected_input_filelist)))

    test_index = list(set(full_index) - set(train_index))

    input_filelist_train = [
                             selected_input_filelist[i] for i in sorted(train_index)
                             ]

    output_filelist_train = [
                             selected_output_filelist[i] for i in sorted(train_index)
                              ]

    input_filelist_test = [
                             selected_input_filelist[i] for i in sorted(test_index)
                             ]

    output_filelist_test = [
                             selected_output_filelist[i] for i in sorted(test_index)
                             ]

    save_stats(input_filelist_train, output_filelist_train, output_filelist_test)

    input_train_mean = torch.load('input_train_mean.pt')
    input_train_std = torch.load('input_train_std.pt')
    output_train_var = torch.load('output_train_var.pt')
    output_test_var = torch.load('output_test_var.pt')

    train_dataset = StarDataset(input_filelist_train, output_filelist_train)
    test_dataset = StarDataset(input_filelist_test, output_filelist_test)

    #kwargs = {'num_workers': 8,
    #              'pin_memory': True} if torch.cuda.is_available() else {}

    torch.manual_seed(processor_id + 1)
    #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(processor_id + 1)
    #test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Loaded data!')

    n_out_pixels_train = len(train_index) * train_loader.dataset[0][1].numel()
    n_out_pixels_test = len(test_index) * test_loader.dataset[0][1].numel()
    
    model = DenseNet(in_channels= 4, out_channels=2,
                blocks=blocks,
                growth_rate=growth_rate,
                init_features=init_features,
                drop_rate=drop_rate,
                bn_size= 4,
                bottleneck= True,
                out_activation='Softplus')

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr,
                 weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-6)

    for epoch in range(1, epochs + 1):
        model.train()
        nll_total = 0.
        mse  = 0.
        for input_features, output_qois in train_loader:
            "normalize each input feature over the 2-D geometry"
            input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            input_features, output_qois = input_features.to(device), output_qois.to(device)
            optimizer.zero_grad()
            output = model(input_features)
            mu, sig = output[:,0,:,:], sp(output[:,1,:,:]) + 1e-6
            loss = torch.sum(nll_loss(mu,sig, output_qois.squeeze(1)))
            loss.backward()
            optimizer.step()
        nll_total += loss.item()
        mse += F.mse_loss(mu, output_qois.squeeze(1), reduction = 'sum').item()

        nll_train = nll_total / n_out_pixels_train
        scheduler.step(nll_train)
        rmse_train = np.sqrt(mse / n_out_pixels_train)
        r2_train = 1 - mse / n_out_pixels_train / output_train_var
        r2_train = r2_train.numpy()

        print(f"Process {processor_id}, epoch: {epoch}, train nll: {nll_train}")
        print(f"Process {processor_id}, epoch: {epoch}, train rmse: {rmse_train}")
        print(f"process {processor_id}, epoch: {epoch}, train r2: {r2_train}")

    with torch.no_grad():
        model.eval()
        nll_total = 0.
        mse = 0.
        for input_features, output_qois in test_loader:
            "normalize each input feature over the 2-D geometry"
            input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            input_features, output_qois = input_features.to(device), output_qois.to(device)
            output = model(input_features)
            mu, sig = output[:,0,:,:], sp(output[:,1,:,:]) + 1e-6
            loss = torch.sum(nll_loss(mu,sig, output_qois.squeeze(1)))
            nll_total += loss.item()
            mse += F.mse_loss(mu, output_qois.squeeze(1), reduction = 'sum').item()

        nll_test = nll_total / n_out_pixels_test
        rmse_test = np.sqrt(mse / n_out_pixels_test)
        r2_test = 1 - mse / n_out_pixels_test / output_test_var
        r2_test = r2_test.numpy()

        print(f"Process {processor_id}, epoch: {epoch}, test nll: {nll_test}")
        print(f"Process {processor_id}, epoch: {epoch}, test rmse: {rmse_test}")
        print(f"Process {processor_id}, epoch: {epoch}, test r2: {r2_test}")
    return rmse_test

hyperparameter_sets = [
    {"lr": 0.0008, "weight_decay": 0.002, "drop_rate": 0.15, "growth_rate": 16, "batch_size": 16, "init_features": 32, "blocks_1_layers": 3, "blocks_2_layers": 4, "blocks_3_layers": 5, "blocks_4_layers": 3, "blocks_5_layers": 3, "num_blocks": 3},  # 1
    {"lr": 0.0007, "weight_decay": 0.003, "drop_rate": 0.17, "growth_rate": 12, "batch_size": 16, "init_features": 16, "blocks_1_layers": 3, "blocks_2_layers": 4, "blocks_3_layers": 5, "blocks_4_layers": 3, "blocks_5_layers": 5, "num_blocks": 5}, # 2
    {"lr": 0.0006, "weight_decay": 0.004, "drop_rate": 0.19, "growth_rate": 24, "batch_size": 12, "init_features": 64, "blocks_1_layers": 4, "blocks_2_layers": 5, "blocks_3_layers": 6, "blocks_4_layers": 4, "blocks_5_layers": 6, "num_blocks": 3},  # 3
    {"lr": 0.0005, "weight_decay": 0.005, "drop_rate": 0.21, "growth_rate": 32, "batch_size": 24, "init_features": 32, "blocks_1_layers": 4, "blocks_2_layers": 4, "blocks_3_layers": 4, "blocks_4_layers": 8, "blocks_5_layers": 7, "num_blocks": 3}, # 4
    {"lr": 0.0004, "weight_decay": 0.001, "drop_rate": 0.23, "growth_rate": 8, "batch_size": 8, "init_features": 16, "blocks_1_layers": 3, "blocks_2_layers": 5, "blocks_3_layers": 7, "blocks_4_layers": 9, "blocks_5_layers": 5, "num_blocks": 3}, # 5
    {"lr": 0.0003, "weight_decay": 0.006, "drop_rate": 0.25, "growth_rate": 16, "batch_size": 48, "init_features": 64, "blocks_1_layers": 5, "blocks_2_layers": 4, "blocks_3_layers": 5, "blocks_4_layers": 3, "blocks_5_layers": 6, "num_blocks": 5},  # 6
    {"lr": 0.0002, "weight_decay": 0.0009, "drop_rate": 0.27, "growth_rate": 24, "batch_size": 8, "init_features": 32, "blocks_1_layers": 4, "blocks_2_layers": 3, "blocks_3_layers": 3, "blocks_4_layers": 5, "blocks_5_layers": 5, "num_blocks": 5}, # 7 
    {"lr": 0.001, "weight_decay": 0.002, "drop_rate": 0.07, "growth_rate": 24, "batch_size": 32, "init_features": 16, "blocks_1_layers": 5, "blocks_2_layers": 5, "blocks_3_layers": 6, "blocks_4_layers": 9, "blocks_5_layers": 7, "num_blocks": 3}, # 8
    {"lr": 0.0009, "weight_decay": 0.0007, "drop_rate": 0.31, "growth_rate": 8, "batch_size": 32, "init_features": 16, "blocks_1_layers": 6, "blocks_2_layers": 7, "blocks_3_layers": 8, "blocks_4_layers": 6, "blocks_5_layers": 4, "num_blocks": 3}, # 9 
    {"lr": 0.002, "weight_decay": 0.0005, "drop_rate": 0.14, "growth_rate": 24, "batch_size": 48, "init_features": 32, "blocks_1_layers": 6, "blocks_2_layers": 4, "blocks_3_layers": 5, "blocks_4_layers": 4, "blocks_5_layers": 3, "num_blocks": 3}, # 10 
]


def run_optimization(base_parameters, processor_id):

    ax_client = AxClient()

    ax_client.create_experiment(
        name="tune_dcnn",  # The name of the experiment.
        parameters=[
            {
                "name": "lr",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [1e-4, 1e-2],  # The bounds for range parameters.
                # "values" The possible values for choice parameters .
                # "value" The fixed value for fixed parameters.
                "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                "log_scale": True,  # Optional, whether to use a log scale for range parameters. Defaults to False.
                # "is_ordered" Optional, a flag for choice parameters.
            },
            {
                "name": "weight_decay",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [1e-4, 1e-2],  # The bounds for range parameters.
                "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                "log_scale": True,  # Optional, whether to use a log scale for range parameters. Defaults to False.
            },
            {
                "name": "drop_rate",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [0, 0.5],  # The bounds for range parameters.
                "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
            },
            {
                "name": "growth_rate",  # The name of the parameter.
                "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
                "values": [8, 12, 16, 24, 32, 48],
                "value_type": "int",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                "is_ordered": True,
            },
            {
                "name": "batch_size",  # The name of the parameter.
                "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
                "values": [8, 12, 16, 24, 32, 48, 64, 128],
                "value_type": "int",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                "is_ordered": True,
            },
            {
                "name": "init_features",  # The name of the parameter.
                "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
                "values": [8, 12, 16, 24, 32, 48, 64, 128],
                "value_type": "int",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                "is_ordered": True,
            },
            #{
             #   "name": "epochs",  # The name of the parameter.
             #   "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
             #   "values": [200, 300, 400, 500],
             #   "value_type": "int",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
             #   "is_ordered": True,
            #},
            {
                "name": "num_blocks",  # The name of the parameter.
                "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
                "values": [3, 5],
                "value_type": "int",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                "is_ordered": True,
            },
            {
                "name": "blocks_1_layers",  # Layers in the first block
                "type": "choice",
                "values": [3, 4, 5, 6, 7, 8, 9],
                "value_type": "int",
                "is_ordered": True,
            },
            {
                "name": "blocks_2_layers",  # Layers in the second block
                "type": "choice",
                "values": [3, 4, 5, 6, 7, 8, 9],
                "value_type": "int",
                "is_ordered": True,
            },
            {
                "name": "blocks_3_layers",  # Layers in the third block
                "type": "choice",
                "values": [3, 4, 5, 6, 7, 8, 9],
                "value_type": "int",
                "is_ordered": True,
            },
            {
                "name": "blocks_4_layers",  # Layers in the third block
                "type": "choice",
                "values": [3, 4, 5, 6, 7, 8, 9],
                "value_type": "int",
                "is_ordered": True,
            },
            {
                "name": "blocks_5_layers",  # Layers in the third block
                "type": "choice",
                "values": [3, 4, 5, 6, 7, 8, 9],
                "value_type": "int",
                "is_ordered": True,
            },
        ],
        objectives={"rmse_test": ObjectiveProperties(minimize=True)},  # The objective name and minimization setting.
    # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
    # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
    )

    ax_client.attach_trial(parameters=base_parameters)

    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)

    ax_client.complete_trial(trial_index=0, raw_data=test_bayesian(baseline_parameters, processor_id))

    
    for i in range(75):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=test_bayesian(parameters, processor_id))

    best_parameters, values = ax_client.get_best_parameters()
    print(f"Best parameters for processor {processor_id} {best_parameters}")
    print(f"Best values {values}")
    df = ax_client.get_trials_data_frame()
    df.to_csv(f"trials_data_process{processor_id}.csv", index=False)
    ax_client.save_to_json_file(f'ax_client_state_processor_{procesor_id}.json')

def main():
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=len(hyperparameter_sets)) as executor:
        # Submit optimization tasks to the executor
        futures = {executor.submit(run_optimization, hyperparams, processor_id): (hyperparams, processor_id) 
                   for processor_id, hyperparams in enumerate(hyperparameter_sets, start = 1)}

        for future in as_completed(futures):
            hyperparams, processor_id = futures[future]
            try:
                best_parameters = future.result()
                print(f"Processor {processor_id}: Best parameters for {hyperparams}: {best_parameters}")
            except Exception as exc:
                print(f"Processor {processor_id}: Hyperparameter set {hyperparams} generated an exception: {exc}")
    end_time = time.time()  # Record end time
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()


