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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import random

from fisher.model_merging.model import clone_model
from fisher.model_merging.data import load_models_regression, load_fishers, create_dataset, load_grads
from fisher.model_merging.merging import merging_models_fisher, merging_models_isotropic
from fisher.model_merging.model import MLP_regression
from fisher.train_regression import train, inference
from fisher.merge_permutation import merging_models_permutation
from model_merging.fisher_regression import compute_fisher_diags, compute_fisher_grads

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif', 'size': 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

##############################################
# Set up dataset
##############################################

def manual_dataset(cfg, overlapping=True, nof=5000, dim=1, k=1, alpha=1.0, beta=0.01):
    if overlapping:
        X = torch.rand(nof, dim)
    else:
        X = torch.zeros(nof, dim)
        X[:, :] = torch.rand(nof, dim) 

    W = alpha*torch.randn(dim)

    f_k = W[0] + X[:,:] @ W[:,None]
    Y = f_k + beta*torch.randn_like(f_k)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size_train, shuffle=True)
    
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size_test, shuffle=False)

    cfg.data.batch_size_test = 1
    test_loader2 = DataLoader(test_dataset, batch_size=cfg.data.batch_size_test, shuffle=False)

    return train_loader, test_loader, test_loader2   

############################################
# Main
############################################

@hydra.main(config_path="./configurations", config_name="regression.yaml")
def main(cfg): 

    dim = cfg.data.dimensions
    n_models = 3
    names = []
    models = []
    
    # train_loader, test_loader, inference_loader = manual_dataset(cfg, dim=dim)
    dataset = create_dataset(cfg)
    train_loader, test_loader = dataset.create_dataloaders()
    #inference_loader = dataset.create_inference_dataloader()
    inference_loader = test_loader

    for i in range(n_models):
        model = MLP_regression(cfg)
        optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        criterion = torch.nn.MSELoss()
        name = "{}regressor_{}.pt".format(
            cfg.data.model_path,
            i
        )

        cfg.train.lr = random.randint(1,3)*cfg.train.lr
        cfg.train.epochs = random.randint(30,50)

        name = train(cfg, name, train_loader, test_loader, model, optimizer, criterion)
        compute_fisher_diags(cfg, name, train_loader)
        compute_fisher_grads(cfg, name, train_loader)
        model.eval()
        models.append(model)
        names.append(name)


    params = models[0].get_trainable_parameters()
    metatheta = torch.nn.utils.parameters_to_vector(params)
    print("Parameters: {}".format(len(metatheta)))
    
    random_model = MLP_regression(cfg)
    x_random, y_random, avg_loss_random = inference(cfg, random_model, inference_loader, criterion)
    print("Random untrained - Average loss {}".format(avg_loss_random))

    for m, model in enumerate(models):
        _, _, avg_loss_model = inference(cfg, model, inference_loader, criterion)
        print("Model {} - Average loss {}".format(m, avg_loss_model)) 

    # FISHER
    output_model = clone_model(models[0], cfg)
    models = load_models_regression(cfg, names)
    fishers = load_fishers(cfg, names)
    fisher_model = merging_models_fisher(output_model, models, fishers)
    x_fisher, y_fisher, avg_loss_fisher = inference(cfg, fisher_model, inference_loader, criterion)
    print("Fisher - Average loss {}".format(avg_loss_fisher)) 

    # ISOTROPIC
    output_model = clone_model(models[0], cfg)
    models = load_models_regression(cfg, names)
    isotropic_model = merging_models_isotropic(output_model, models)
    x_isotropic, y_isotropic, avg_loss_isotropic = inference(cfg, isotropic_model, inference_loader, criterion)
    print("Isotropic - Average loss {}".format(avg_loss_isotropic))

    # PERMUTATION
    models = load_models_regression(cfg, names)
    grads = load_grads(cfg, names)
    metamodel = isotropic_model 
    metamodel = fisher_model
    perm_model = merging_models_permutation(cfg, metamodel, models, grads, inference_loader, criterion, plot=True)
    x_perm, y_perm, avg_loss_permutation = inference(cfg, perm_model, inference_loader, criterion)
    print("Ours (after) - Average loss {}".format(avg_loss_permutation)) 

    if dim == 1:
        palette = ['#264653', '#C2FCF7', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
        labels = ['Data', 'Random', 'Isotropic', 'Fisher', 'Perm', 'Models']
        x_data = []
        y_data = []

        for x,y in inference_loader:
            x_data.append(np.array(x.flatten()))
            y_data.append(np.array(y.flatten()))

        m = 50
        plt.scatter(x_data, y_data, marker="+", c=palette[0])
        plt.scatter(x_random[:m], y_random[:m], marker=".", c=palette[1])
        plt.scatter(x_isotropic[:m], y_isotropic[:m], marker=".", c=palette[2])
        plt.scatter(x_fisher[:m], y_fisher[:m], marker=".", c=palette[3])
        plt.scatter(x_perm[:m], y_perm[:m], marker=".", c=palette[4])

        for x, model in enumerate(models):
            x_model, y_model, _ = inference(cfg, model, inference_loader, criterion)
            plt.scatter(x_model[:m], y_model[:m], marker=">", c=palette[5])

        plt.legend(labels=labels, fontsize=8)

        plt.xlabel('$X$')
        plt.ylabel('$Y$')
        plt.show()

        
if __name__ == "__main__": 
    main()