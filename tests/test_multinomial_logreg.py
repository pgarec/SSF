
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import tqdm
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal as Normal

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


parser = argparse.ArgumentParser()
parser.add_argument('--overlapping', '-over', type=bool, default=False)
parser.add_argument('--num_k', '-k', type=int, default=2)
parser.add_argument('--post_samples', '-s', type=int, default=20)
parser.add_argument('--nof', '-n', type=int, default=1000)
parser.add_argument('--epochs', '-e', type=int, default=2000)
parser.add_argument('--beta', '-b', type=float, default=0.1)
parser.add_argument('--alpha', '-a', type=float, default=1.0)
parser.add_argument('--plot', '-p', type=bool, default=True)
args = parser.parse_args()

############################################
# DATA - 3 Linear Classification problems
############################################

# Set up true curves and models
if args.overlapping:
    X = torch.rand(args.nof,args.num_k)
else:
    X = torch.zeros(args.nof,args.num_k)
    for k in range(args.num_k):
        X[:,k] = (1/args.num_k)*torch.rand(args.nof,1).flatten() + k*(1/args.num_k)

W = torch.randn(2,args.num_k)

if args.plot:
    plt.figure()

datasets = []
for k in range(args.num_k):
    f_k = W[0,k] + W[1,k]*X[:,k]
    y_k = f_k + args.beta*torch.randn_like(f_k)

    datasets.append({'x': X[:,k], 'y': y_k})

    if args.plot:
        plt.plot(X[:,k], f_k, c=palette[k])
        plt.scatter(X[:,k], y_k, s=8, c=palette[k], alpha=0.25)

if args.plot:
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.show()

############################################
# Models - Multinomial Logistic Regression
############################################