# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
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

from model_merging.model import MLP 
from model_merging.data import MNIST, load_models
from model_merging.data import load_fisher

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def load_fishers(cfg):
    fishers = []

    for model_name in cfg.models:
        path = cfg.data.fisher_path + cfg.models[model_name]
        fisher = load_fisher(path)
        fishers.append(fisher)

    return fishers


# def shannon_loss(model, fishers, optimizer, cfg):
#     log_p_masked = -0.5*torch.log(v_pred) - 0.5*np.log(2*np.pi) - (0.5*(x_p[:,:m] - m_pred)**2 / v_pred)


@hydra.main(config_path="./configurations", config_name="data.yaml")
def main(cfg):
    fishers = load_fishers(cfg)
    models = load_models(cfg)
    metamodel = MLP(cfg)
    optimizer = optim.SGD(
        metamodel.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum
    )
    # loss = shannon_loss(metamodel, models, fishers)

    # for epoch in range(cfg.train.epochs):
    #     loss.backward()
    #     optimizer.step()
    #     train_loss += loss

if __name__ == "__main__": 
    main()