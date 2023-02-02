
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import tqdm
import torch
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


K = 3
S = 20
N = 1000
epochs = 3000
beta = 0.1
alpha = 1.0
plot = True
############################################
# DATA - 3 Linear Regression problems
############################################

# Set up true curves and models
X = torch.rand(N,K)
W = torch.randn(2,K)

if plot:
    plt.figure()

datasets = []
for k in range(K):
    f_k = W[0,k] + W[1,k]*X[:,k]
    y_k = f_k + beta*torch.randn_like(f_k)

    datasets.append({'x': X[:,k], 'y': y_k})

    if plot:
        plt.plot(X[:,k], f_k, c=palette[k])
        plt.scatter(X[:,k], y_k, s=8, c=palette[k], alpha=0.25)

if plot:
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.show()

############################################
# Models - Bayesian Linear Regression
############################################

if plot:
    plt.figure()

models = []
for k in range(K):
    x_k = datasets[k]['x']
    y_k = datasets[k]['y']

    phi_k = torch.ones(N,2)
    phi_k[:,1] = x_k

    iS_k = alpha*torch.eye(2) + beta * phi_k.T @ phi_k
    S_k = torch.inverse(iS_k)
    m_k = beta*S_k @ phi_k.T @ y_k

    models.append({'m': m_k.unsqueeze(1), 'iS': iS_k})

    if plot:
        
        f_m =  m_k[0] + m_k[1]*x_k
        plt.plot(x_k, f_m, c=palette[k], ls='--')
        plt.scatter(X[:,k], y_k, s=8, c=palette[k], alpha=0.25)
        model_k = Normal(loc=m_k, covariance_matrix=S_k)

        W_s = model_k.rsample((1,S))[0,:,:].T
        for s in range(S):
            f_s = W_s[0,s] + W_s[1,s]*x_k
            plt.plot(x_k, f_s, c=palette[k], alpha=0.2, lw=1.0)

if plot:
    plt.title('Bayesian Linear Regression -- Posteriors')
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    # plt.show()

# i think there is a tiny mistake on the equations

############################################
# Models from models
############################################

class MetaPosterior(torch.nn.Module):
    def __init__(self, models):
        super(MetaPosterior, self).__init__()

        self.models = models
        self.m = torch.nn.Parameter(torch.randn(2,1), requires_grad=True)  # variational meta-posterior: mean parameter
        self.L = torch.nn.Parameter(torch.eye(2), requires_grad=True)  # variational meta-posterior: covariance

    def expectation(self, meta_m, meta_S):
        E = 0.0

        # Expectation of k ensembles --
        for k, model_k in enumerate(self.models):

            m_k = model_k['m']
            iS_k = model_k['iS']

            # Expectation on terms -- E[log_p()] and E[log_q()]
            E_log_q = -torch.trace(iS_k.mm(meta_S)) - (meta_m - m_k).t().mm(iS_k).mm(meta_m - m_k) - torch.logdet(2*np.pi*S_k)


            E_log_p = -torch.trace((1/alpha)*torch.eye(2).mm(meta_S)) - meta_m.t().mm((1/alpha)*torch.eye(2)).mm(meta_m) - torch.logdet(2*np.pi*alpha*torch.eye(2))

            # General Expectation -- E[sum_k E[log_q_k] - E[log_p_k]]
            E += 0.5*(E_log_q - E_log_p) #+ model_k.logZ


        return E

    def forward(self):

        meta_m = self.m
        meta_L = torch.tril(self.L)
        meta_S = torch.mm(meta_L, meta_L.t())

        q_dist = Normal(meta_m.flatten(), meta_S)
        p_dist = Normal(torch.zeros(2), alpha*torch.eye(2))

        # Expectation --
        expectation = self.expectation(meta_m, meta_S)

        # KL divergence --
        kl = kl_divergence(q_dist, p_dist)
        # Calls ELBO
        elbo = expectation - kl
        return -elbo
    
meta_model = MetaPosterior(models)
# optimizer = torch.optim.SGD([{'params':meta_model.m, 'lr':1e-3},{'params':meta_model.L,'lr':1e-6}], lr=1e-4, momentum=0.9)
optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
pbar = tqdm.trange(epochs)


elbo_its = []
for it in pbar:
    elbo_it = meta_model()    # Forward pass -> computes ELBO
    optimizer.zero_grad()
    elbo_it.backward()      # Backward pass <- computes gradients
    optimizer.step()
    elbo_its.append(-meta_model().item())
    pbar.set_description(f'[Loss: {-meta_model().item():.3f}')

    # print('  \__ elbo =', meta_model().item())


if plot:
    # plt.figure()
    S_meta = 5*S
    meta_m = meta_model.m.detach().flatten()
    meta_L = torch.tril(meta_model.L.detach())
    meta_S = torch.mm(meta_L, meta_L.t())

    f_meta =  meta_model.m[0] + meta_model.m[1]*x_k
    plt.plot(x_k, f_meta.detach().numpy(), c=meta_color, ls='--')
    meta_model = Normal(loc=meta_m, covariance_matrix=meta_S)

    W_s = meta_model.rsample((1,S_meta))[0,:,:].T
    for s in range(S_meta):
        f_s = W_s[0,s] + W_s[1,s]*x_k
        plt.plot(x_k, f_s, c=meta_color, alpha=0.2, lw=1.0)

    plt.show()