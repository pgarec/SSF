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

    s_loss = l - (len(models)-1)*prior

    return s_loss

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
        names.append(name)

        name = train(cfg, name, train_loader, test_loader, model, optimizer, criterion)
        print("compute_fisher_diags")
        compute_fisher_diags(cfg, name)
        compute_fisher_grads(cfg, name)
        model.eval()
        models.append(model)

    # FISHER
    fishers = load_fishers(cfg, names)
    fisher_model = merging_models_fisher(cfg, models, fishers)
    inference(cfg, fisher_model, test_loader, criterion)

    # ISOTROPIC
    models = load_models(cfg)
    isotropic_model = merging_models_isotropic(cfg, models)
    inference(cfg, isotropic_model, test_loader, criterion)

    # PERMUTATION
    models = load_models(cfg)
    random_model = MLP_regression(cfg)
    metamodel = isotropic_model # siempre inicializar en isotropic -- decision que yo tomaria
    # metamodel = fisher_model
    # metamodel = MLP(cfg)

    avg_loss = inference(cfg, random_model, test_loader, criterion)
    print("Random untrained - Average loss {}".format(avg_loss))
    avg_loss = inference(cfg, isotropic_model, test_loader, criterion)
    print("Isotropic - Average loss {}".format(avg_loss))
    avg_loss = inference(cfg, fisher_model, test_loader, criterion)
    print("Fisher - Average loss {}".format(avg_loss)) 

    perm_model = merging_models_permutation(cfg, random_model, models, grads, test_loader, criterion)
    cfg.train.plot = False

    avg_loss = inference(cfg, perm_model, test_loader, criterion)
    print("Ours (after) - Average loss {}".format(avg_loss))  


    
if __name__ == "__main__": 
    main()