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
import torch.nn as nn
import math
import os 
import pickle

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

parser = argparse.ArgumentParser()
parser.add_argument('--overlapping', '-over', type=bool, default=True)
parser.add_argument('--dim', '-d', type=int, default=1)
parser.add_argument('--mask', '-m', type=int, default=7)
parser.add_argument('--num_k', '-k', type=int, default=2)
parser.add_argument('--post_samples', '-s', type=int, default=10)
parser.add_argument('--nof', '-n', type=int, default=1000)
parser.add_argument('--epochs', '-e', type=int, default=2000)
parser.add_argument('--beta', '-b', type=float, default=0.1)
parser.add_argument('--alpha', '-a', type=float, default=1.0)
parser.add_argument('--plot', '-plot', type=bool, default=True)
parser.add_argument('--max_perm', '-p', type=int, default=10)
args = parser.parse_args()

############################################
# DATA - 3 Linear Regression problems
############################################

seed = 444
np.random.seed(444)
torch.manual_seed(444)

# Set up true curves and models
if args.overlapping:
    X = torch.rand(args.nof, args.dim, args.num_k)
else:
    X = torch.zeros(args.nof, args.dim, args.num_k)
    for k in range(args.num_k):
        X[:, :, k] = (1/args.num_k)*torch.rand(args.nof,args.dim) + k*(1/args.num_k)

W = args.alpha*torch.randn(args.dim, args.num_k)

if args.plot and args.dim<2:
    plt.figure()

datasets = []
for k in range(args.num_k):
    f_k = W[0,k] + X[:,:,k] @ W[:,None,k]
    # f_k = W[0,0] + W[1,0]*X[:,k]
    y_k = f_k + args.beta*torch.randn_like(f_k)

    datasets.append({'x': X[:,:,k], 'y': y_k})

    if args.plot and args.dim<2:
        plt.plot(X[:,0,k], f_k, c=palette[k])
        plt.scatter(X[:,0,k], y_k, s=8, c=palette[k], alpha=0.25)

if args.plot and args.dim<2:
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    # plt.show()

############################################
# Models - Bayesian Linear Regression
############################################

X_all = torch.zeros(args.nof*args.num_k, args.dim)
Y_all = torch.zeros(args.nof*args.num_k, args.dim)
models = []
for k in range(args.num_k):
    x_k = datasets[k]['x']
    y_k = datasets[k]['y']

    # Stacking all data together for later
    X_all[k*args.nof:(k+1)*args.nof,:] = x_k
    Y_all[k*args.nof:(k+1)*args.nof] = y_k

    phi_k = torch.ones(args.nof,args.dim+1)
    phi_k[:,1:] = x_k

    iS_k = args.alpha*torch.eye(args.dim+1) + args.beta * phi_k.T @ phi_k
    S_k = torch.inverse(iS_k)

    m_k = args.beta*S_k @ phi_k.T @ y_k

    models.append({'m': m_k, 'iS': iS_k, 'S': S_k})

############################################
# True Meta Posterior
############################################

phi = torch.ones(args.nof*args.num_k, args.dim+1)
phi[:,1:] = X_all

iS = args.alpha*torch.eye(args.dim+1) + args.beta * phi.T @ phi
S = torch.inverse(iS)
m = args.beta*S @ phi.T @ Y_all
true_theta = m

############################################
# Models from models
############################################

def active_set_permutation(x, W):
    """ Description:    Does a random permutation of data and selects a subset
    Input:          Data observations X (NxD)
    Return:         Active Set X_A and X_rest / X_A U X_rest = X
    """ 
    permutation = torch.randperm(x.size()[1])

    W_perm = W[permutation]
    x_perm = x[:, permutation]

    return x_perm, W_perm, permutation


def compute_gradients(k):
    S = torch.tensor([0.5])
    
    return (X[:,:,k] @ X[:,:,k].T) / (S)

# Meta Posterior
class MetaPosterior(torch.nn.Module):
    def __init__(self, models, grads, datasets, args):
        super(MetaPosterior, self).__init__()

        self.models = models
        self.grads = grads
        self.meta_theta = torch.nn.Parameter(args.alpha*torch.randn(args.dim+1,1), requires_grad=True)  # MAP of the meta-posterior
        self.datasets = datasets

    def forward(self):
        prior = Normal(loc=torch.zeros(args.dim+1), covariance_matrix=args.alpha*torch.eye(args.dim+1))
        loss = (1 - (1/args.num_k))*prior.log_prob(self.meta_theta.squeeze())
        for a in range(args.dim):
            # m = a+1
            m = args.mask

            loss_pred = 0.0
            for p in range(args.max_perm):
                for k, model_k in enumerate(models):
                    perm = torch.randperm(args.dim+1)
                    grad = grads[k]
                    m_k = model_k['m']
                    theta = self.meta_theta[perm]

                    theta_r = theta[m:]
                    P_mr = grad[perm[:m],:][:,perm[m:]]
                    P_mm = grad[perm[:m],:][:,perm[:m]] + 0.01 * torch.eye(m)
                    # iP_mm = 1/P_mm

                    m_pred = m_k[perm[:m]] - torch.linalg.solve(P_mm, P_mr) @ (theta_r - m_k[perm[m:]])
                    p_pred = torch.diagonal(P_mm)

                    log_p_masked = - 0.5*np.log(2*torch.tensor([math.pi])) + 0.5*torch.log(p_pred)  - (0.5* p_pred *(theta[:m] - m_pred)**2)
                    loss_pred += log_p_masked.sum(1)

            loss_pred = loss_pred/(args.max_perm * m * args.num_k)
            loss += loss_pred.sum()

        return -loss 

############################################
# Definition of Meta-Model and ELBO fitting
############################################
grads = [compute_gradients(i) for i, model in enumerate(models)]
meta_model = MetaPosterior(models, grads, datasets, args)
optimizer = torch.optim.SGD(params=meta_model.parameters(), lr=1e-4, momentum=0.9)
# optimizer = torch.optim.SGD([{'params':meta_model.m, 'lr':1e-3},{'params':meta_model.L,'lr':1e-6}], lr=1e-4, momentum=0.9)
# optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-2)
pbar = tqdm.trange(args.epochs)

map_mse = []
elbo_its = []
for it in pbar:
    elbo_it = meta_model()    # Forward pass -> computes ELBO
    optimizer.zero_grad()
    elbo_it.backward()      # Backward pass <- computes gradients
    optimizer.step()
    elbo_its.append(-meta_model().item())
    map_mse.append(torch.sum((true_theta - meta_model.meta_theta.detach())**2)/(args.dim+1))
    pbar.set_description(f'[Loss: {-meta_model().item():.3f}')
    # print('  \__ elbo =', meta_model().item())


if args.plot:
    directory = "./images/{}_d{}_m{}_{}epochs_seed{}/".format("LR_FISHER", args.dim, args.mask, args.max_perm, seed)
   
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.figure()
    plt.plot(elbo_its, c=meta_color, ls='-', alpha=0.7)
    plt.title(r'Training the Meta Posterior')
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Loss')
    plt.show()

    plt.figure()
    plt.plot(torch.arange(len(map_mse)), [map_mse[-1]]*(len(map_mse)), '--', color='k', alpha=0.7)
    plt.plot(map_mse, c=meta_color, ls='-', alpha=0.7)
    plt.plot(len(map_mse)-1, map_mse[-1],c=meta_color, marker='x')
    plt.ylim(0,max(map_mse))
    plt.xlim(0,len(map_mse)+10)
    plt.xlabel(r'Epochs')
    plt.ylabel(r'MSE')
    plt.title(r'Difference -- true \textsc{map} vs \textsc{meta-map}')
    plt.show()

    with open('{}map_mse'.format(directory), 'wb') as f:
        pickle.dump(map_mse, f)

    with open('{}elbo_its'.format(directory), 'wb') as f:
        pickle.dump(elbo_its, f)


