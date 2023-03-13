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

from fisher.model_merging.data import load_models, load_fishers, create_dataset
from fisher.model_merging.merging import merging_models_fisher_subsets
from fisher.model_merging.model import MLP_regression
from fisher.train_regression import train_regression, inference

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

############################################
# Shannon loss
############################################

def logprob_multivariate_gaussian(x, mu, var):
    var += 0.1
    n = x.shape[0]    

    return - n * torch.log((1/var)) -0.5 * n * np.log(2*torch.pi) -0.5 * (x-mu)**2 * var


def shannon_loss(metamodel, models, fishers):
    params = metamodel.get_trainable_parameters()
    metamean = nn.utils.parameters_to_vector(params)
    prior_mean = torch.zeros(metamean.shape[0])
    prior_cov = 10e-4 * torch.ones(metamean.shape[0])

    prior = logprob_multivariate_gaussian(metamean, prior_mean, prior_cov).sum()
    l = 0

    for m in range(len(models)):
        model = models[m]
        mean = nn.utils.parameters_to_vector(model.parameters())
        fisher = nn.utils.parameters_to_vector(fishers[m])
        l += logprob_multivariate_gaussian(metamean, mean, fisher).sum()

    #s_loss = (1/((m+1)*metamean.shape[0]))*l - (len(models)-1)*prior
    s_loss = l - (len(models)-1)*prior

    return s_loss

############################################
# Feature extraction
############################################

def extract_features(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = []

    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            out = model.feature_map(x.to(device))
            features.append((out, y))

    return features

############################################
# T-SNE
############################################

def tsne(cfg, features):
    features_tsne = []
    labels_tsne = []
    for m in features:
        for x,y in m:
            features_tsne.append(np.array(x))
            labels_tsne.append(np.array(y))


    test_features = np.concatenate(features_tsne)
    test_labels = np.concatenate(labels_tsne)

    tsne = TSNE(n_components=2, perplexity=40, random_state=0, n_iter=300)
    test_features_tsne = tsne.fit_transform(test_features)
    
    cmap = plt.get_cmap('tab10', 10)  # Choose a discrete colormap with 4 colors
    plt.figure(figsize=(10, 5))
    plt.scatter(test_features_tsne[:, 0], test_features_tsne[:, 1], c=test_labels, cmap=cmap)
    plt.title('Test set t-SNE embeddings')
    plt.colorbar()
    plt.show()

############################################
# Main
############################################

@hydra.main(config_path="./configurations", config_name="regression.yaml")
def main(cfg):
    model = MLP_regression(cfg)
    dataset = create_dataset(cfg)
    train_loader, test_loader = dataset.create_dataloaders()
    optimizer = optim.SGD(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
    criterion = torch.nn.MSELoss()

    name = "prova"

    train_regression(cfg, name, train_loader, test_loader, model, optimizer, criterion)




    

if __name__ == "__main__": 
    main()