
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)


import torch
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


K = 5
S = 50
N = 1000
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
    print(W[0,k], W[1,k])
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

    models.append({'m': m_k, 'iS': iS_k})

    if plot:
        
        f_m =  m_k[0] + m_k[1]*x_k
        plt.plot(x_k, f_m, c=palette[k], ls='--')
        plt.scatter(X[:,k], y_k, s=8, c=palette[k], alpha=0.25)
        model_k = MultivariateNormal(loc=m_k, covariance_matrix=S_k)

        W_s = model_k.rsample((1,S))[0,:,:].T
        for s in range(S):
            f_s = W_s[0,s] + W_s[1,s]*x_k
            plt.plot(x_k, f_s, c=palette[k], alpha=0.2, lw=1.0)

if plot:
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.show()

# i think there is a tiny mistake on the equations

############################################
# Models from models
############################################