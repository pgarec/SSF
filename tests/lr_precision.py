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
import os 
import pickle

font = {'family' : 'serif',
        'size'   : 20}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

color_palette_1 = ['#335c67','#fff3b0','#e09f3e','#9e2a2b','#540b0e']
color_palette_2 = ['#177e89','#084c61','#db3a34','#ef8354','#323031']
color_palette_3 = ['#bce784','#5dd39e','#348aa7','#525274','#513b56']
color_palette_4 = ['#002642','#840032','#e59500','#e5dada','#02040e']
color_palette_5 = ['#202c39','#283845','#b8b08d','#f2d449','#f29559']

palette_red = ["#03071e","#370617","#6a040f","#9d0208","#d00000","#dc2f02","#e85d04","#f48c06","#faa307","#ffba08"]
palette_blue = ["#012a4a","#013a63","#01497c","#014f86","#2a6f97","#2c7da0","#468faf","#61a5c2","#89c2d9","#a9d6e5"]
palette_green = ['#99e2b4','#88d4ab','#78c6a3','#67b99a','#56ab91','#469d89','#358f80','#248277','#14746f','#036666']
palette_pink = ["#ea698b","#d55d92","#c05299","#ac46a1","#973aa8","#822faf","#6d23b6","#6411ad","#571089","#47126b"]
palette_super_red = ["#641220","#6e1423","#85182a","#a11d33","#a71e34","#b21e35","#bd1f36","#c71f37","#da1e37","#e01e37"]

palette = color_palette_2
meta_color = 'r'
parser = argparse.ArgumentParser()
parser.add_argument('--overlapping', '-over', type=bool, default=True)
parser.add_argument('--dim', '-d', type=int, default=1)
parser.add_argument('--mask', '-m', type=int, default=7)
parser.add_argument('--num_k', '-k', type=int, default=2)
parser.add_argument('--post_samples', '-s', type=int, default=10)
parser.add_argument('--nof', '-n', type=int, default=1000)
parser.add_argument('--epochs', '-e', type=int, default=200)
parser.add_argument('--beta', '-b', type=float, default=0.1)
parser.add_argument('--alpha', '-a', type=float, default=1.0)
parser.add_argument('--plot', '-plot', type=bool, default=True)
parser.add_argument('--max_perm', '-p', type=int, default=10)
args = parser.parse_args()

############################################
# DATA - 3 Linear Regression problems
############################################

seed = 440
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

# i think there is a tiny mistake on the equations

############################################
# True Meta Posterior Fisher
############################################

phi = torch.ones(args.nof*args.num_k, args.dim+1)
phi[:,1:] = X_all

iS = args.alpha*torch.eye(args.dim+1) + args.beta * phi.T @ phi
S = torch.inverse(iS)
m = args.beta*S @ phi.T @ Y_all
true_theta = m

############################################
# Models from models Fisher
############################################

def compute_fisher(k):
    S = torch.tensor([0.5])
    N = X[:,:,k].shape[0]

    return (X[:,:,k].T @ X[:,:,k]) / (S)

# Meta Posterior
class MetaPosteriorFisher(torch.nn.Module):
    def __init__(self, models, fishers, datasets, args):
        super(MetaPosteriorFisher, self).__init__()

        self.models = models
        self.fishers = fishers
        self.meta_theta = torch.nn.Parameter(args.alpha*torch.randn(args.dim+1,1), requires_grad=True)  # MAP of the meta-posterior
        self.datasets = datasets

    def forward(self):
        prior = Normal(loc=torch.zeros(args.dim+1), covariance_matrix=args.alpha*torch.eye(args.dim+1))
        loss = (1 - (1/args.num_k))*prior.log_prob(self.meta_theta.squeeze())
        for a in range(args.dim):
            m = args.mask

            loss_pred = 0.0
            for p in range(args.max_perm):
                for k, model_k in enumerate(models):
                    perm = torch.randperm(args.dim)
                    fisher = self.fishers[k]
                    m_k = model_k['m']
                    theta = self.meta_theta[perm]

                    theta_r = theta[m:]
                    P_mr = fisher[perm[:m],:][:,perm[m:]]
                    P_mm = fisher[perm[:m],:][:,perm[:m]] 
                    iP_mm = torch.inverse(P_mm)
                    m_pred = m_k[perm[:m]] - iP_mm @ P_mr @ (theta_r - m_k[perm[m:]])
                    p_pred = torch.diagonal(P_mm)

                    log_p_masked = - 0.5*np.log(2*torch.tensor([math.pi])) + 0.5*torch.log(p_pred)  - (0.5* p_pred *(theta[:m] - m_pred)**2)
                    loss_pred += log_p_masked.sum(1)

            loss_pred = loss_pred/(args.max_perm * m * args.num_k)
            loss += loss_pred.sum()

        return -loss 

############################################
# Models from models
############################################

# Meta Posterior
class MetaPosterior(torch.nn.Module):
    def __init__(self, models, args):
        super(MetaPosterior, self).__init__()

        self.models = models
        self.meta_theta = torch.nn.Parameter(args.alpha*torch.randn(args.dim+1,1), requires_grad=True)  # MAP of the meta-posterior

    def forward(self):
        prior = Normal(loc=torch.zeros(args.dim+1), covariance_matrix=args.alpha*torch.eye(args.dim+1))
        # loss = (args.num_k - 1)*prior.log_prob(self.meta_theta.squeeze())
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
                    iS_k = model_k['iS']
                    S_k = model_k['S']

                    theta_r = theta[m:]
                    P_mr = iS_k[perm[:m],:][:,perm[m:]]
                    P_mm = iS_k[perm[:m],:][:,perm[:m]]
                    iP_mm = torch.inverse(P_mm)

                    m_pred = m_k[perm[:m]] - iP_mm @ P_mr @ (theta_r - m_k[perm[m:]])
                    p_pred = torch.diagonal(P_mm)

                    log_p_masked = - 0.5*np.log(2*torch.tensor([math.pi])) + 0.5*torch.log(p_pred)  - (0.5* p_pred *(theta[:m] - m_pred)**2)
                    loss_pred += log_p_masked.sum(1)

            loss_pred = loss_pred/(args.max_perm * m * args.num_k)
            loss += loss_pred.sum()

        return -loss # minimization 

############################################
# Definition of Meta-Model and ELBO fitting 
############################################

seed = 440
torch.manual_seed(seed)
np.random.seed(seed)

meta_model = MetaPosterior(models, args)
optimizer = torch.optim.SGD(params=meta_model.parameters(), lr=1e-4, momentum=0.9)
pbar = tqdm.trange(args.epochs)

map_mse1 = []
elbo_its1 = []
for it in pbar:
    elbo_it = meta_model()    # Forward pass -> computes ELBO
    optimizer.zero_grad()
    elbo_it.backward()      # Backward pass <- computes gradients
    optimizer.step()
    elbo_its1.append(-meta_model().item())
    map_mse1.append(torch.sum((true_theta - meta_model.meta_theta.detach())**2)/(args.dim+1))
    pbar.set_description(f'[Loss: {-meta_model().item():.3f}')
    # print('  \__ elbo =', meta_model().item())


seed = 441
torch.manual_seed(seed)
np.random.seed(seed)
meta_model = MetaPosterior(models, args)
optimizer = torch.optim.SGD(params=meta_model.parameters(), lr=1e-4, momentum=0.9)
pbar = tqdm.trange(args.epochs)

map_mse2 = []
elbo_its2 = []
for it in pbar:
    elbo_it = meta_model()    # Forward pass -> computes ELBO
    optimizer.zero_grad()
    elbo_it.backward()      # Backward pass <- computes gradients
    optimizer.step()
    elbo_its2.append(-meta_model().item())
    map_mse2.append(torch.sum((true_theta - meta_model.meta_theta.detach())**2)/(args.dim+1))
    pbar.set_description(f'[Loss: {-meta_model().item():.3f}')
    # print('  \__ elbo =', meta_model().item())


seed = 442
torch.manual_seed(seed)
np.random.seed(seed)

meta_model = MetaPosterior(models, args)
optimizer = torch.optim.SGD(params=meta_model.parameters(), lr=1e-4, momentum=0.9)
pbar = tqdm.trange(args.epochs)

map_mse3 = []
elbo_its3 = []
for it in pbar:
    elbo_it = meta_model()    # Forward pass -> computes ELBO
    optimizer.zero_grad()
    elbo_it.backward()      # Backward pass <- computes gradients
    optimizer.step()
    elbo_its3.append(-meta_model().item())
    map_mse3.append(torch.sum((true_theta - meta_model.meta_theta.detach())**2)/(args.dim+1))
    pbar.set_description(f'[Loss: {-meta_model().item():.3f}')
    # print('  \__ elbo =', meta_model().item())


if args.plot:


    ###################################################
    with open('./fisher/map_mse_fisher442', 'rb') as f:
        map_mse_f1 = pickle.load(f)

    with open('./fisher/map_mse_fisher441', 'rb') as f:
        map_mse_f2 = pickle.load(f)
    
    with open('./fisher/map_mse_fisher442', 'rb') as f:
        map_mse_f3 = pickle.load(f)
    
    with open('./fisher/elbo_fisher442', 'rb') as f:
        elbo_its_f1 = pickle.load(f)
    
    with open('./fisher/elbo_fisher441', 'rb') as f:
        elbo_its_f2 = pickle.load(f)
    
    with open('./fisher/elbo_fisher442', 'rb') as f:
        elbo_its_f3 = pickle.load(f)

    plt.figure()
    plt.plot(elbo_its1,  lw=4.5, alpha=0.4, color=palette_pink[3+1], label="Precision")
    plt.plot(elbo_its2,  lw=4.5, alpha=0.4, color=palette_pink[3+2])
    plt.plot(elbo_its3,  lw=4.5, alpha=0.4, color=palette_pink[3+3])
    plt.plot(elbo_its_f1,  lw=4.5, alpha=0.4, color=palette_blue[3+1], label="Fisher")
    plt.plot(elbo_its_f2,  lw=4.5, alpha=0.4, color=palette_blue[3+2])
    plt.plot(elbo_its_f3,  lw=4.5, alpha=0.4, color=palette_blue[3+3])
    plt.xlim(0,len(elbo_its2)-1)

    plt.title(r'Training the Meta Posterior')
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Loss')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    plt.plot(torch.arange(len(map_mse1)), [map_mse1[-1]]*(len(map_mse1)), '--', color='k', alpha=0.7)
    plt.plot(len(map_mse1)-1, map_mse1[-1],c=meta_color, marker='x')

    plt.plot(map_mse1, lw=4.5, alpha=0.4, color=palette_pink[3+1], label="Precision")
    plt.plot(map_mse2, lw=4.5, alpha=0.4, color=palette_pink[3+2])
    plt.plot(map_mse3, lw=4.5, alpha=0.4, color=palette_pink[3+3])
    plt.plot(map_mse_f1, lw=4.5, alpha=0.4, color=palette_blue[3+1], label="Fisher")
    plt.plot(map_mse_f2, lw=4.5, alpha=0.4, color=palette_blue[3+2])
    plt.plot(map_mse_f3, lw=4.5, alpha=0.4, color=palette_blue[3+3])
    
    plt.ylim(0,max(map_mse1))
    plt.xlim(0,len(map_mse1)-1)

    plt.xlabel(r'Epochs')
    plt.ylabel(r'MSE')
    plt.title(r'Difference -- true \textsc{map} vs \textsc{meta-map}')
    plt.legend(loc="upper right")

    plt.show()