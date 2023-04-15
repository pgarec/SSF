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

from src.model_merging.model import MLP
from src.model_merging.data import load_models, load_fishers, create_dataset
from src.model_merging.merging import merging_models_fisher_subsets
from src.train import inference

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

@hydra.main(config_path="./configurations", config_name="data.yaml")
def main(cfg):
    fishers = load_fishers(cfg)
    models = load_models(cfg)
    
    # FISHER
    n_classes = cfg.data.n_classes*cfg.data.n_models
    output_model = MLP(cfg)
    fisher_model = merging_models_fisher_subsets(output_model, models, fishers, n_classes)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = create_dataset(cfg)
    test_loader = dataset.create_inference_dataloader()
    # inference(cfg, fisher_model, test_loader, criterion)

    # ISOTROPIC
    # cfg.data.n_classes = 2
    # isotropic_model = merging_models_fisher_subsets(cfg, models)
    # # inference(cfg, isotropic_model, test_loader, criterion)

    # # SHANNON 
    # cfg.data.n_classes = 2
    # metamodel = merging_models_isotropic(cfg, models)
    for name, param in fisher_model.named_parameters():
        if not 'clf' in name:
            param.requires_grad = False

    optimizer = optim.SGD(fisher_model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)
    pbar = tqdm.trange(100)

    shannon_losses = []

    for it in pbar:
        optimizer.zero_grad()
        l = shannon_loss(fisher_model, models, fishers)
        l.backward()      # Backward pass <- computes gradients
        optimizer.step()
        shannon_losses.append(l.item())
        pbar.set_description(f'[Loss: {l.item():.3f}')

    # # MODEL 0
    # cfg.data.n_classes = 2
    # cfg.data.classes = [0,1]
    # dataset = create_dataset(cfg)
    # test_loader = dataset.create_inference_dataloader()
    # # inference(cfg, models[0], test_loader, criterion)

    # # MODEL 1
    # cfg.data.n_classes = 2
    # cfg.data.classes = [2,3]
    # dataset = create_dataset(cfg)
    # test_loader = dataset.create_inference_dataloader()
    # # inference(cfg, models[1], test_loader, criterion)

    # # FEATURE EXTRACTION
    # cfg.data.classes = [0,1]
    # dataset = create_dataset(cfg)
    # test_loader = dataset.create_inference_dataloader()
    # features1 = extract_features(models[0], test_loader)

    # cfg.data.classes = [2,3]
    # dataset = create_dataset(cfg)
    # test_loader = dataset.create_inference_dataloader()
    # features2 = extract_features(models[1], test_loader)

    cfg.data.classes = [0,1,2,3]
    dataset = create_dataset(cfg)
    test_loader = dataset.create_inference_dataloader()
    features3 = extract_features(fisher_model, test_loader)

    # T-SNE
    # tsne(cfg, [features])


if __name__ == "__main__": 
    main()