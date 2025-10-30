#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:52:18 2020

@author: yang.liu
"""

import argparse
import torch
import json
import random
from pprint import pprint
from utils.misc import mkdirs

# always uses cuda if avaliable

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Dense Convolutional Networks for 3-Dimensional turbulent viscosity prediction')
        self.add_argument('--case_dir', type=str, default="./experiments", help='directory to save experiments')
        self.add_argument('--post', action='store_true', default=False, help='post training analysis')

        # network
        self.add_argument('--in_channels', type=int, default=4, help='number of input features')
        self.add_argument('--blocks', nargs='+', default = [3,4,5],help='list of number of layers in each dense block')
#        self.add_argument('--blocks', type=list, default=[3,6,3], help='list of number of layers in each dense block')
        self.add_argument('--growth_rate', type=int, default=16, help='number of output feature maps of each conv layer within each dense block')
        self.add_argument('--init_features', type=int, default=32, help='number of initial features after the first conv layer')
        self.add_argument('--drop_rate', type=float, default=0.15, help='dropout rate')
        self.add_argument('--bn_size', type=int, default=4, help='bottleneck size: bn_size * growth_rate')
        self.add_argument('--bottleneck', action='store_true', default=False, help='enables bottleneck design in the dense blocks')

        # data
#        self.add_argument('--data_dir', type=str, default="./dataset", help='directory to dataset')

        # training
        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
        self.add_argument('--lr', type=float, default=8e-4, help='learnign rate')
        # self.add_argument('--lr-scheduler', type=str, default='plateau', help="scheduler, plateau or step")
        self.add_argument('--weight_decay', type=float, default=2e-3, help="weight decay")
        self.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
#        self.add_argument('--test_batch_size', type=int, default=128, help='input batch size for testing (default: 100)')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')

        # logging
        self.add_argument('--ckpt_epoch', type=int, default=None, help='which epoch of checkpoints to be loaded in post mode')
        self.add_argument('--ckpt_freq', type=int, default=100, help='how many epochs to wait before saving model')
        self.add_argument('--log_freq', type=int, default=1, help='how many epochs to wait before logging training status')
#        self.add_argument('--plot_freq', type=int, default=100, help='how many epochs to wait before plotting test output')
#        self.add_argument('--plot_fn', type=str, default='contourf', choices=['contourf', 'imshow'], help='plotting method')


    def parse(self):
        args = self.parse_args()
        
        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        args.run_dir = args.case_dir + '/' \
            + '/seed{}'.format(args.seed
            )
        args.ckpt_dir = args.run_dir + '/checkpoints'
        mkdirs([args.run_dir, args.ckpt_dir])

        assert args.epochs % args.ckpt_freq == 0, 'epochs must'\
            'be dividable by ckpt_freq'

        print('Arguments:')
        pprint(vars(args))

        if not args.post:
            with open(args.run_dir + "/args.txt", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)
        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
