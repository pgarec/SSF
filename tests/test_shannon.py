# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CogSys Section  ---  (pgare@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal as Normal
import hydra
import torch.optim as optim
import torch.nn.functional as F
from laplace import Laplace
from laplace.utils import ModuleNameSubnetMask
import torch.nn.functional as F
import tqdm
import torchbnn as bnn
import torch.nn as nn
 
from model_merging.model import MLP 
from model_merging.data import MNIST, load_models
from model_merging.data import load_fishers

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

############################################
# Meta Model
############################################

def compute_log_prior(model):
        log_prior = 0.0
        for module in model.modules():
            if isinstance(module, bnn.BayesLinear):
                log_prior += module.log_prior().sum()
        return log_prior


class BayesianMetaPosterior(torch.nn.Module):
    def __init__(self, cfg, models, fishers):
        super(BayesianMetaPosterior, self).__init__()

        self.models = models
        self.cfg = cfg
        self.fishers = fishers
    
        self.feature_map = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=cfg.data.image_shape, out_features=cfg.train.hidden_dim),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=cfg.train.hidden_dim, out_features=cfg.train.hidden_dim),
            nn.ReLU()
        )

        self.clf = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=cfg.train.hidden_dim, out_features=cfg.data.n_classes)
        )
    
    def get_trainable_parameters(self):
        return [param.data for param in self.parameters() if param.requires_grad]

    def shannon_loss(self):
        prior = torch.distributions.Normal(0, 0.1)  # Define a normal prior with mean 0 and standard deviation 0.1
        params = self.get_trainable_parameters()
        metamean = nn.utils.parameters_to_vector(params)
        l = - prior.log_prob(torch.tensor(metamean)).sum() * (len(self.models)-1)

        for m in range(len(self.models)):
            model = self.models[m]
            mean = nn.utils.parameters_to_vector(model.parameters())
            f = [k.detach() for k in self.fishers[m]]
            fisher = nn.utils.parameters_to_vector(f)
            cov = torch.diag(fisher)
            posterior = torch.distributions.MultivariateNormal(mean.detach(), torch.inverse(cov))
            l += posterior.log_prob(mean.detach()).sum().detach()

        return l


@hydra.main(config_path="./configurations", config_name="data.yaml")
def main(cfg):
    fishers = load_fishers(cfg)
    models = load_models(cfg)
    optimizer = optim.Adam(
        metamodel.parameters(), lr=cfg.train.lr
    )
    pbar = tqdm.trange(cfg.train.epochs)
    metamodel = BayesianMetaPosterior(cfg, models, fishers)    # Forward pass -> computes ELBO

    elbo_its = []
    for it in pbar:
        optimizer.zero_grad()
        l = metamodel.shannon_loss()
        l.backward()      # Backward pass <- computes gradients
        optimizer.step()
        # elbo_its.append(().item())
        # pbar.set_description(f'[Loss: {-meta_model().item():.3f}')

if __name__ == "__main__": 
    main()