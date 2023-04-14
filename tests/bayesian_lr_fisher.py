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

from fisher.model_merging.fisher_regression import _compute_exact_grads_for_batch

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
parser.add_argument('--mask', '-m', type=int, default=1)
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

# i think there is a tiny mistake on the equations

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


def compute_gradients(model, dataset):
    m = model['m'].flatten()
    # S = model['S']
    S = torch.tensor([0.5])
    grads = torch.zeros(m.flatten().shape, requires_grad=False)
    grad_samples = 1000
    n_examples = 1
    m.requires_grad = True

    for x,y in zip(dataset['x'], dataset['y']):
        print(n_examples)
        n_examples += 1     
        f_m = m[0] + x @ m[1:] 
        log_p = -0.5*torch.log(S) - 0.5*np.log(2*np.pi) - (0.5*(y - f_m)**2 / S)
        loss = log_p.sum()
        loss.backward()
        # f_m.backward()
        grad = m.grad
        grads = [x+y for (x,y) in zip(grads, grad)]

        if grad_samples != -1 and n_examples > grad_samples:
            break

    m.requires_grad = False

    for i, grad in enumerate(grads):
        grads[i] = grad / n_examples

    return grads

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
        # loss = (args.num_k - 1)*prior.log_prob(self.meta_theta.squeeze())
        loss = (1 - (1/args.num_k))*prior.log_prob(self.meta_theta.squeeze())
        for a in range(args.dim):
            # m = a+1
            m = 2

            # average over permutations -- 
            loss_pred = 0.0
            for p in range(args.max_perm):
                for k, model_k in enumerate(models):
                    perm = torch.randperm(args.dim+1)
                    grad = nn.utils.parameters_to_vector(grads[k])
                    m_k = model_k['m']
                    theta = self.meta_theta[perm]

                    # iS_k_mr = torch.outer(grad[perm[:m]], grad[perm[m:]])
                    # i_K1 = torch.outer(grad[perm[:m]],grad[perm[:m]])

                    # m_pred = m_k[perm[:m]] - torch.inverse(i_K1) @ (iS_k_mr @ (self.meta_theta[perm[m:]] - m_k[perm[m:]]))
                    # v_pred = torch.diagonal(torch.inverse(i_K1))

                    # log_p_masked = - 0.5*np.log(2*torch.tensor([math.pi])) - 0.5*torch.log(v_pred)  - (0.5*(self.meta_theta[perm[:m]] - m_pred)**2) / v_pred
                    # loss_pred += log_p_masked.sum(1)

                    log_p_masked = 0
                    for i in range(m):
                        
                        theta_r = theta[m:]
                        P_mr = torch.outer(grad[perm[i]].unsqueeze(0), grad[perm[m:]])
                        P_mm = grad[perm[i]]
                       
                        m_pred = m_k[perm[i]] - (1/P_mm) @ P_mr @ (theta_r - m_k[perm[m:]])
                        p_pred = P_mm

                        log_p_masked += - 0.5*np.log(2*torch.tensor([math.pi])) + 0.5*torch.log(p_pred)  - (0.5* p_pred *(theta[i] - m_pred)**2)
       

            loss_pred = loss_pred/(args.max_perm * m * args.num_k)
            loss += loss_pred.sum()
            # masked_loss += loss_pred.sum()/args.dim

        return -loss # minimization 

############################################
# Definition of Meta-Model and ELBO fitting
############################################
grads = [compute_gradients(model, datasets[k]) for k, model in enumerate(models)]
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

# if args.plot:
#     # plt.figure()
#     x_meta = torch.rand(args.nof,args.dim)
#     meta_w = meta_model.meta_theta.detach()

#     f_meta =  meta_w[0] + x_meta @ meta_w[1:]
#     plt.plot(x_meta, f_meta.detach().numpy(), c=meta_color, ls='-', lw=0.5, alpha=0.5)
#     plt.show()


# check on test data
