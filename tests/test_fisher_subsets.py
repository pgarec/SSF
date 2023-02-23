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
 
from fisher.model_merging.model import MLP 
from fisher.model_merging.data import MNIST, load_models
from fisher.model_merging.data import load_fishers
from fisher.model_merging.merging import merging_models_fisher_subsets
from fisher.model_merging.evaluation import evaluate_metamodel, evaluate_minimodels, plot


@hydra.main(config_path="./configurations", config_name="subset.yaml")
def main(cfg):
    criterion = torch.nn.CrossEntropyLoss()
    fishers = load_fishers(cfg)
    models = load_models(cfg)
    merged_model = merging_models_fisher_subsets(cfg, models, fishers)

    cfg.data.batch_size_train = 1
    cfg.data.batch_size_test = 1
    cfg.data.n_classes = 6
    dataset = MNIST(cfg)
    _, test_loader_metamodel = dataset.create_dataloaders()
    avg_loss, count = evaluate_metamodel(cfg, merged_model, criterion, test_loader_metamodel)

    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.digits))]
    print("Avg_loss merged model: {}".format(avg_loss))
    
    plt.bar(np.arange(len(cfg.data.digits)), avg_loss)
    plt.xlabel("Number of classes")
    plt.ylabel("Average Test Loss")
    plt.xticks(np.arange(len(cfg.data.digits)))
    plt.show()

    # MODEL 1
    cfg.data.n_classes = 3
    cfg.data.digits = [0,1,2]
    dataset = MNIST(cfg)
    _, test_loader_metamodel = dataset.create_dataloaders()
    avg_loss, count = evaluate_metamodel(cfg, models[0], criterion, test_loader_metamodel)
    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.digits))]
    print("Avg_loss merged model: {}".format(avg_loss))
    
    plt.bar(np.arange(len(cfg.data.digits)), avg_loss)
    plt.xlabel("Number of classes")
    plt.ylabel("Average Test Loss")
    plt.xticks(np.arange(len(cfg.data.digits)))
    plt.show()

    # MODEL 1
    cfg.data.n_classes = 3
    cfg.data.digits = [3,4,5]
    dataset = MNIST(cfg)
    _, test_loader_metamodel = dataset.create_dataloaders()
    avg_loss, count = evaluate_metamodel(cfg, models[1], criterion, test_loader_metamodel)
    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.digits))]
    print("Avg_loss merged model: {}".format(avg_loss))
    
    plt.bar(np.arange(len(cfg.data.digits)), avg_loss)
    plt.xlabel("Number of classes")
    plt.ylabel("Average Test Loss")
    plt.xticks(np.arange(len(cfg.data.digits)))
    plt.show()

if __name__ == "__main__":
    main()