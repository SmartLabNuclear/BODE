#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:22:45 2020

@author: yang.liu
"""

import torch
from torch.utils.data import Dataset
        
class StarDataset(Dataset):
    def __init__(self, input_list, output_list):
        'initilization only, not loading data'
        self.input_list = input_list
        self.output_list = output_list
        if len(self.input_list) != len(self.output_list) :
            raise ValueError('length of input list and output list must be the same!')
            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_list)

    def __getitem__(self, index):
        'check if the input/output pair set are correct'
        if self.input_list[index][-7:-3] != self.output_list[index][-7:-3]: # the last four characters from the input filename and the output filename
            print(self.input_list[index])
            print(self.output_list[index])
            raise ValueError('input and output pair are inconsistent!') 
        
        # load data
        input_features = torch.load(self.input_list[index])
        A_idx = [0,2,3,4]  # remove velocity j component from inputs
 #       input_features = input_features[A_idx,int(output_qois.size()[1]/2),:,:]   # cut one slice from the data
        input_features = input_features[A_idx,int(input_features.size()[1]/2),:,:]   # cut one slice from the data

        output_qois = torch.load(self.output_list[index])
        output_qois = output_qois[0,int(output_qois.size()[1]/2),:,:].unsqueeze(0)
        
        # convert to float type
        input_features = input_features.type(torch.float)
        output_qois = output_qois.type(torch.float)

        return input_features, output_qois
