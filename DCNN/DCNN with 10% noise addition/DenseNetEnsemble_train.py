#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:00:02 2020

@author: yang.liu
"""
import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
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
from save_stats import save_stats
import scipy.ndimage
#args.train_dir = args.run_dir + "/training"
#args.pred_dir = args.train_dir + "/predictions"
#mkdirs([args.train_dir, args.pred_dir])

# load data

# load data
input_filelist = sorted(glob('./sample/input_*.pt'))
output_filelist = sorted(glob('./sample/output_*.pt'))


random.seed(args.seed)
" select data files for training, randomly "

num_files_to_select = int(1 * len(input_filelist))
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

kwargs = {'num_workers': 8,
              'pin_memory': True} if torch.cuda.is_available() else {}

torch.manual_seed(args.seed)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

torch.manual_seed(args.seed)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

print('Loaded data!')

n_out_pixels_train = len(train_index) * train_loader.dataset[0][1].numel()
n_out_pixels_test = len(test_index) * test_loader.dataset[0][1].numel()

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

# initialize model
torch.manual_seed(args.seed)
model = DenseNet(in_channels=args.in_channels, out_channels=2,
                blocks=args.blocks,
                growth_rate=args.growth_rate,
                init_features=args.init_features,
                drop_rate=args.drop_rate,
                bn_size=args.bn_size,
                bottleneck=args.bottleneck,
                out_activation='Softplus')
print(model)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model.to(device)


# define training process
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                 weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-6)

" Proper scoring rule using negative log likelihood scoring rule"

nll_loss = lambda mu, sigma, y: torch.log(sigma)/2 + ((y-mu)**2)/(2*sigma)
sp = torch.nn.Softplus()

def train(epoch):
    model.train()
    nll_total = 0.
    mse  = 0.
    for input_features, output_qois in train_loader:

        "normalize each input feature over the 2-D geometry"
        input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features, output_qois = input_features.to(device), output_qois.to(device)
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, size=output_qois.shape)
        smooth_noise_factor = scipy.ndimage.gaussian_filter(noise, sigma =12)
        smoothed_noise_factor_std = np.std(smooth_noise_factor)
        rescaling_factor = (noise_std/smoothed_noise_factor_std)
        smooth_noise_factor_rescaled = smooth_noise_factor * rescaling_factor
        smooth_noise_factor_tensor = torch.from_numpy(smooth_noise_factor_rescaled).to(output_qois.device)
        smooth_noise_factor_tensor = smooth_noise_factor_tensor.to(output_qois.dtype)
        smooth_noise = smooth_noise_factor_tensor * output_qois
        output_qois_noisy = torch.clamp(smooth_noise + output_qois, min = 0).to(output_qois.device)

        optimizer.zero_grad()
        output = model(input_features)
        mu, sig = output[:,0,:,:], sp(output[:,1,:,:]) + 1e-6
        loss = torch.sum(nll_loss(mu,sig, output_qois_noisy.squeeze(1)))
        loss.backward()

        optimizer.step()
        nll_total += loss.item()
        mse += F.mse_loss(mu, output_qois_noisy.squeeze(1), reduction = 'sum').item()

    nll_train = nll_total / n_out_pixels_train
    scheduler.step(nll_train)

    rmse_train = np.sqrt(mse / n_out_pixels_train)
    r2_train = 1 - mse /n_out_pixels_train/ output_train_var
    r2_train = r2_train.numpy()

    print("epoch: {}, train nll: {:.6f}".format(epoch, nll_train))
    print("epoch: {}, train r2: {:.6f}".format(epoch, r2_train))
    print("epoch: {}, train rmse: {:.6f}".format(epoch, rmse_train))

    if epoch % args.log_freq == 0:
        logger['r2_train'].append(r2_train)
        logger['nll_train'].append(nll_train)
        logger['rmse_train'].append(rmse_train)
        f = open(args.run_dir + '/' + 'nll_train.pkl',"wb")
        pickle.dump(logger['nll_train'],f)
        f.close()
        f = open(args.run_dir + '/' + 'r2_train.pkl',"wb")
        pickle.dump(logger['r2_train'],f)
        f.close()
        f = open(args.run_dir + '/' + 'rmse_train.pkl',"wb")
        pickle.dump(logger['rmse_train'],f)
        f.close()
    # save model
    if epoch % args.ckpt_freq == 0:
        tic2 = time()
        print("Trained 100 epochs with using {} seconds".format(tic2 - tic))
        torch.save(model.module.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))

def test(epoch):
    model.eval()
    nll_total = 0.
    mse = 0.
    for input_features, output_qois in test_loader:

        "normalize each input feature over the 2-D geometry"
        input_features -= input_train_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features /= input_train_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        input_features, output_qois = input_features.to(device), output_qois.to(device)
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, size=output_qois.shape)
        smooth_noise_factor = scipy.ndimage.gaussian_filter(noise, sigma =12)
        smoothed_noise_factor_std = np.std(smooth_noise_factor)
        rescaling_factor = (noise_std/smoothed_noise_factor_std)
        smooth_noise_factor_rescaled = smooth_noise_factor * rescaling_factor
        smooth_noise_factor_tensor = torch.from_numpy(smooth_noise_factor_rescaled).to(output_qois.device)
        smooth_noise_factor_tensor = smooth_noise_factor_tensor.to(output_qois.dtype)
        smooth_noise = smooth_noise_factor_tensor * output_qois
        output_qois_noisy = torch.clamp(smooth_noise + output_qois, min = 0).to(output_qois.device)


        output = model(input_features)

        mu, sig = output[:,0,:,:], sp(output[:,1,:,:]) + 1e-6
        loss = torch.sum(nll_loss(mu,sig, output_qois_noisy.squeeze(1)))

        nll_total += loss.item()
        mse += F.mse_loss(mu, output_qois_noisy.squeeze(1), reduction = 'sum').item()


    nll_test = nll_total / n_out_pixels_test

    rmse_test = np.sqrt(mse / n_out_pixels_test)

    r2_test = 1 - mse/n_out_pixels_test/output_test_var
    r2_test = r2_test.numpy()
    print("epoch: {}, test nll: {:.6f}".format(epoch, nll_test))
    print("epoch: {}, test r2: {:.6f}".format(epoch, r2_test))
    print("epoch: {}, test rmse: {:.6f}".format(epoch, rmse_test))

    if epoch % args.log_freq == 0:
        logger['r2_test'].append(r2_test)
        logger['nll_test'].append(nll_test)
        logger['rmse_test'].append(rmse_test)
        f = open(args.run_dir + '/' + 'nll_test.pkl',"wb")
        pickle.dump(logger['nll_test'],f)
        f.close()
        f = open(args.run_dir + '/' + 'r2_test.pkl',"wb")
        pickle.dump(logger['r2_test'],f)
        f.close()
        f = open(args.run_dir + '/' + 'rmse_test.pkl',"wb")
        pickle.dump(logger['rmse_test'],f)
        f.close()

print('Start training........................................................')
tic = time()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    with torch.no_grad():
        test(epoch)
tic2 = time()
print("Finished training {} epochs  using {} seconds"
      .format(args.epochs,  tic2 - tic))

args.training_time = tic2 - tic
#args.n_params, args.n_layers = model._num_parameters_convlayers()
#print(args.n_params)

with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)
