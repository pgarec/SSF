# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CogSys Section  ---  (pgare@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from src.model_merging.data import load_models, load_fishers, create_dataset
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.train import inference
from src.model_merging.model import clone_model
import pickle
import os
import matplotlib.pyplot as plt
import omegaconf
import numpy as np
import seaborn as sns
from tests.clf_nn_pinwheel import Model
from tests.clf_nn_pinwheel import clone_model, evaluate_model
from src.model_merging.curvature import compute_fisher_diagonals, compute_gradients
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.merge_permutation import merging_models_permutation
from src.model_merging.datasets.pinwheel import make_pinwheel_data

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


# CONFIGURATION
cfg = omegaconf.OmegaConf.load('./configurations/perm_pinwheel.yaml')
seed = cfg.train.torch_seed

if seed > -1:
    np.random.seed(seed)
    torch.manual_seed(seed)

num_clusters = 5        # number of clusters in pinwheel data
samples_per_cluster = 1000  # number of samples per cluster in pinwheel
K = 15                     # number of components in mixture model
N = 2                      # number of latent dimensions
P = 2                      # number of observation dimensions
H = 16
plot = False
batch_data = True


def train_pinwheel_models():
    data, labels = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

    # define train and validation 
    X_train = data[:350]
    X_valid = data[350:400]
    X_test = data[400:]

    y_train = labels[:350].astype('int32')
    y_valid = labels[350:400].astype('int32')
    y_test = labels[400:].astype('int32')

    one_hot_yvtest = np.zeros((y_test.size, y_test.max() + 1))
    one_hot_yvtest[np.arange(y_test.size), y_test.reshape(-1)] = 1

    X_train = torch.from_numpy(X_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    X_test = torch.from_numpy(X_test).float()

    y_train = torch.from_numpy(y_train).long().reshape(-1)
    y_valid = torch.from_numpy(y_valid).long().reshape(-1)
    y_test = torch.from_numpy(y_test).long().reshape(-1)

    if batch_data:
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True)

    if plot:
        plt.scatter(X_train[:,0], X_train[:,1], s=40, c=y_train, cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
        plt.title("Train data")
        plt.show()

        plt.scatter(X_valid[:,0], X_valid[:,1], s=40, c=y_valid, cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
        plt.title("Validation data")
        plt.show()

    num_features = X_train.shape[-1]
    print(num_features)
    num_output = num_clusters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    n_models = 3
    max_epoch = 100

    for m in range(n_models):
        model = Model(num_features, H, num_output, seed)
        lr = cfg.train.lr*((m+1)*0.5)
        optimizer = torch.optim.SGD(model.parameters(),  lr=lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
        criterion = nn.CrossEntropyLoss(reduction='sum')

        max = max_epoch*(m+1)

        for epoch in range(max):
            train_loss = 0
            for _, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                out = model(x.to(device))
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                train_loss += loss
            print(
                f"Epoch [{epoch + 1}/{max}], Training Loss: {train_loss/len(train_loader):.4f}"
            )

            val_loss = 0
            with torch.no_grad():
                model.eval()
                for _, (x, y) in enumerate(val_loader):
                    out = model(x.to(device))
                    loss = criterion(out, y)
                    val_loss += loss

                print(
                    f"Epoch [{epoch + 1}/{max}], Validation Loss: {val_loss/len(val_loader):.4f}"
                )
        
        models.append(model)

    return models, train_loader, val_loader


def generate_plots_m(cfg, directory):
    models, train_loader, val_loader = train_pinwheel_models()
    fishers = [compute_fisher_diagonals(m, train_loader, num_clusters) for m in models]
    criterion = torch.nn.CrossEntropyLoss()
    dataset = create_dataset(cfg)
    test_loader = dataset.create_inference_dataloader()
  
    # FISHER
    models = load_models(cfg)
    output_model = clone_model(models[0], cfg)
    fisher_model = merging_models_fisher(output_model, models, fishers)
    avg_loss_fisher = inference(cfg, fisher_model, test_loader, criterion)

    # ISOTROPIC
    models = load_models(cfg)
    output_model = clone_model(models[0], cfg)
    isotropic_model = merging_models_isotropic(output_model, models)
    avg_loss_isotropic = inference(cfg, isotropic_model, test_loader, criterion)
   
    path_inference = '{}inference_loss'.format(directory)
    path_permutation = '{}perm_loss'.format(directory)

    plt.subplot(2,1,1)
    plt.xlabel('Steps')
    plt.ylabel('Permutation loss')

    for m in [25,50,100,200]:
        path = directory.format(m)
        path_permutation = '{}perm_loss'.format(path)
        
        with open(path_permutation, 'rb') as f:
            perm_losses = pickle.load(f)

        plt.plot(perm_losses)
        break
    
    plt.subplot(2,1,2)
    plt.xlabel('Steps')
    plt.ylabel('Test loss')

    for m in [25,50,100,200]:
        path = directory.format(m)
        path_inference = '{}inference_loss'.format(path)

        with open(path_inference, 'rb') as f:
            inference_loss = pickle.load(f)
            plt.plot(np.log(inference_loss))

    # plt.axhline(avg_loss_fisher, color='b', linestyle='--')
    # plt.text(0.05, avg_loss_fisher + 0.1, f'Fisher loss', color='b', fontsize=10)
    # plt.axhline(avg_loss_isotropic, color='r', linestyle='--')
    # plt.text(0.05, avg_loss_isotropic + 0.1, f'Isotropic loss', color='r', fontsize=10)
    plt.show()
    plt.savefig('{}plot.png'.format(path))

if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load('./configurations/perm_mnist.yaml')
    directory = "./images/PINWHEEL_MLP_m{}_200000epochs_seed40/"

    generate_plots_m(cfg, directory)