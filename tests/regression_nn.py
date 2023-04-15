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

from src.model_merging.model import clone_model
from src.model_merging.data import load_models_regression, load_fishers, create_dataset, load_grads
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.model_merging.model import MLP_regression
from src.train_regression import train, inference
from src.merge_permutation import merging_models_permutation, merging_models_weight_permutation
from model_merging.fisher_regression import compute_and_store_fisher_diagonals, compute_and_store_gradients
from model_merging.permutation import compute_permutations_init, l2_permutation

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
        compute_and_store_fisher_diagonals(cfg, name, train_loader)
        compute_and_store_gradients(cfg, name, train_loader)
        model.eval()
        models.append(model)
        names.append(name)


    params = models[0].get_trainable_parameters()
    metatheta = torch.nn.utils.parameters_to_vector(params)
    print("Parameters: {}".format(len(metatheta)))
    
    random_model = MLP_regression(cfg)
    x_random, y_random, avg_loss_random = inference(random_model, inference_loader, criterion)
    print("Random untrained - Average loss {}".format(avg_loss_random))

    for m, model in enumerate(models):
        _, _, avg_loss_model = inference(model, inference_loader, criterion)
        print("Model {} - Average loss {}".format(m, avg_loss_model)) 

    # FISHER
    output_model = clone_model(models[0], cfg)
    models = load_models_regression(cfg, names)
    fishers = load_fishers(cfg, names)
    fisher_model = merging_models_fisher(output_model, models, fishers)
    x_fisher, y_fisher, avg_loss_fisher = inference(fisher_model, inference_loader, criterion)
    print("Fisher - Average loss {}".format(avg_loss_fisher)) 

    # ISOTROPIC
    output_model = clone_model(models[0], cfg)
    models = load_models_regression(cfg, names)
    isotropic_model = merging_models_isotropic(output_model, models)
    x_isotropic, y_isotropic, avg_loss_isotropic = inference(isotropic_model, inference_loader, criterion)
    print("Isotropic - Average loss {}".format(avg_loss_isotropic))

    # PERMUTATION
    models = load_models_regression(cfg, names)
    grads = load_grads(cfg, names)
    # metamodel = merging_models_isotropic(output_model, models)
    metamodel = MLP_regression(cfg)
    perm_model = merging_models_permutation(cfg, metamodel, models, grads, inference_loader, criterion, plot=True)
    x_perm, y_perm, avg_loss_permutation = inference(perm_model, inference_loader, criterion)
    print("Permutation - Average loss {}".format(avg_loss_permutation)) 

    # WEIGHT SYMMETRIES
    # models = load_models_regression(cfg, names)
    # grads = load_grads(cfg, names)
    # permutations = compute_permutations_init(models, cfg.data.layer_weight_permutation, cfg.data.weight_permutations)
    # # metamodel = merging_models_isotropic(output_model, models)
    # metamodel = MLP_regression(cfg)
    # wperm_model = merging_models_weight_permutation(cfg, metamodel, models, permutations, grads, inference_loader, criterion, plot=True)
    # x_wperm, y_wperm, avg_loss_wpermutation = inference(wperm_model, inference_loader, criterion)
    # print("Weight permutation - Average loss {}".format(avg_loss_wpermutation)) 

    if dim == 1:
        palette = ['#264653', '#C2FCF7', '#2a9d8f', '#e9c46a', '#FE6244']#,'#00E5E8']
        labels = ['Data', 'Random', 'Isotropic', 'Fisher', 'Perm']#,'Wperm']
        x_data = []
        y_data = []

        for x,y in train_loader:
            x_data.append(np.array(x).flatten())
            y_data.append(np.array(y).flatten())

        x_data = np.array(x_data).flatten()
        y_data = np.array(y_data).flatten()

        m = 50
        sorted_indices = np.argsort(x_random[:m])

        plt.scatter(x_data, y_data, marker="+", c=palette[0])
        plt.plot(x_random[:m][sorted_indices], y_random[:m][sorted_indices], c=palette[1])
        plt.plot(x_isotropic[:m][sorted_indices], y_isotropic[:m][sorted_indices], c=palette[2])
        plt.plot(x_fisher[:m][sorted_indices], y_fisher[:m][sorted_indices], c=palette[3])
        plt.plot(x_perm[:m][sorted_indices], y_perm[:m][sorted_indices], c=palette[4])

        for x, model in enumerate(models):
            x_model, y_model, _ = inference(model, inference_loader, criterion)
            labels.append("Model {}".format(x))
            plt.plot(x_model[:m][sorted_indices], y_model[:m][sorted_indices], c="#E11299")

        plt.legend(labels=labels, fontsize=8)
        plt.xlabel('$X$')
        plt.ylabel('$Y$')
        plt.show()

        
if __name__ == "__main__": 
    main()