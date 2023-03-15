# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CogSys Section  ---  (pgare@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import matplotlib.pyplot as plt
import hydra
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE

from fisher.model_merging.data import load_models_regression, load_fishers, create_dataset, load_grads
from fisher.model_merging.merging import merging_models_fisher, merging_models_isotropic
from fisher.model_merging.model import MLP_regression
from fisher.train_regression import train, inference
from fisher.merge_permutation import merging_models_permutation
from model_merging.fisher_regression import compute_fisher_diags, compute_fisher_grads

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

############################################
# Main
############################################

@hydra.main(config_path="./configurations", config_name="regression.yaml")
def main(cfg): 

    n_models = 3
    names = []
    models = []
    dataset = create_dataset(cfg)
    train_loader, test_loader = dataset.create_dataloaders()

    for i in range(n_models):
        model = MLP_regression(cfg)
        optimizer = optim.SGD(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
            )
        criterion = torch.nn.MSELoss()
        name = "{}{}_model{}.pt".format(
            cfg.data.model_path,
            cfg.data.dataset,
            i
        )
        print(name)

        name = train(cfg, name, train_loader, test_loader, model, optimizer, criterion)
        print("compute_fisher_diags")
        compute_fisher_diags(cfg, name)
        compute_fisher_grads(cfg, name)
        model.eval()
        models.append(model)
        names.append(name)
    
    test_loader = dataset.create_inference_dataloader()

    # FISHER
    models = load_models_regression(cfg, names)
    fishers = load_fishers(cfg, names)
    fisher_model = merging_models_fisher(cfg, models, fishers)

    # ISOTROPIC
    models = load_models_regression(cfg, names)
    isotropic_model = merging_models_isotropic(cfg, models)

    # PERMUTATION
    models = load_models_regression(cfg, names)
    random_model = MLP_regression(cfg)
    grads = load_grads(cfg, names)
    metamodel = isotropic_model # siempre inicializar en isotropic -- decision que yo tomaria
    # metamodel = fisher_model
    # metamodel = MLP(cfg)

    y_random, avg_loss_random = inference(cfg, random_model, test_loader, criterion)
    print("Random untrained - Average loss {}".format(avg_loss_random))
    y_isotropic, avg_loss_isotropic = inference(cfg, isotropic_model, test_loader, criterion)
    print("Isotropic - Average loss {}".format(avg_loss_isotropic))
    y_fisher, avg_loss_fisher = inference(cfg, fisher_model, test_loader, criterion)
    print("Fisher - Average loss {}".format(avg_loss_fisher)) 

    perm_model = merging_models_permutation(cfg, random_model, models, grads, test_loader, criterion)
    cfg.train.plot = False

    y_perm, avg_loss_permutation = inference(cfg, perm_model, test_loader, criterion)
    print("Ours (after) - Average loss {}".format(avg_loss_permutation)) 

    values = [avg_loss_random, avg_loss_isotropic, avg_loss_fisher, avg_loss_permutation]
    labels = ["Random", "Isotropic", "Fisher", "Perm"]

    # plt.bar(labels, values)
    # plt.xlabel("Type of merging")
    # plt.ylabel("Average Test Loss")
    # plt.show()

    palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    x_data = []
    y_data = []

    for x,y in test_loader:
        x_data.append(x)
        y_data.append(y)

    plt.scatter(x, y, marker="+", c=palette[0])
    plt.scatter(x, np.array(y_random[0].flatten()), marker=".", c=palette[1])
    plt.scatter(x, np.array(y_isotropic[0].flatten()), marker=".", c=palette[2])
    plt.scatter(x, np.array(y_fisher[0].flatten()), marker=".", c=palette[3])
    plt.scatter(x, np.array(y_perm[0].flatten()), marker=".", c=palette[4])

    # plt.scatter(X[:,k], y_k, s=8, c=palette[k], alpha=0.25)
    
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.show()

    
if __name__ == "__main__": 
    main()