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
import math
import pickle
import os 

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
parser.add_argument('--mask', '-m', type=int, default=2)
parser.add_argument('--num_k', '-k', type=int, default=2)
parser.add_argument('--post_samples', '-s', type=int, default=10)
parser.add_argument('--nof', '-n', type=int, default=1000)
parser.add_argument('--epochs', '-e', type=int, default=1000)
parser.add_argument('--beta', '-b', type=float, default=0.1)
parser.add_argument('--alpha', '-a', type=float, default=1.0)
parser.add_argument('--plot', '-plot', type=bool, default=True)
parser.add_argument('--max_perm', '-p', type=int, default=10)
args = parser.parse_args()

############################################
# DATA - 3 Linear Regression problems
############################################

seed = 444
torch.manual_seed(seed)
np.random.seed(seed)

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
    plt.show()

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

# Meta Posterior
class MetaPosterior(torch.nn.Module):
    def __init__(self, models, args):
        super(MetaPosterior, self).__init__()

        self.models = models
        self.meta_theta = torch.nn.Parameter(args.alpha*torch.randn(args.dim+1,1), requires_grad=True)  # MAP of the meta-posterior

    def forward(self):
        prior = Normal(loc=torch.zeros(args.dim+1), covariance_matrix=args.alpha*torch.eye(args.dim+1))
        loss = (1 - (1/args.num_k))*prior.log_prob(self.meta_theta.squeeze())

        for a in range(args.dim):
            # m = a+1
            m = args.mask
            # average over permutations -- 
            loss_pred = 0.0
            for p in range(args.max_perm):
                for k, model_k in enumerate(models):
                    perm = torch.randperm(args.dim+1)

                    theta = self.meta_theta[perm]
                    m_k = model_k['m']
                    S_k = model_k['S']

                    # Computation // forgetting other extra elements
                    theta_r = theta[m:]
                    S_mm = S_k[perm[:m],:][:,perm[:m]]
                    S_mr = S_k[perm[:m],:][:,perm[m:]]
                    iS_rr = torch.inverse(S_k[perm[m:],:][:,perm[m:]])

                    m_pred = m_k[perm[:m]] + S_mr @ iS_rr @ (theta_r - m_k[perm[m:]])
                    v_pred = torch.diagonal(S_mm - S_mr @ iS_rr @ S_mr.T)                    

                    log_p_masked = - 0.5*np.log(2*torch.tensor([math.pi])) - 0.5*torch.log(v_pred)  - (0.5*(theta[:m] - m_pred)**2) / v_pred
                    loss_pred += log_p_masked.sum(1)

            loss_pred = loss_pred/(args.max_perm * m * args.num_k)
            loss += loss_pred.sum()
        
        return -loss # minimization 

############################################
# Definition of Meta-Model and ELBO fitting
############################################
meta_model = MetaPosterior(models, args)
optimizer = torch.optim.SGD(params=meta_model.parameters(), lr=1e-4, momentum=0.9)
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
    directory = "./images/{}_d{}_m{}_{}epochs_seed{}/".format("LR_COVARIANCE", args.dim, args.mask, args.max_perm, seed)
   
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

