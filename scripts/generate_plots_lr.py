# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CogSys Section  ---  (pgare@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import matplotlib.pyplot as plt
from src.model_merging.data import load_models, load_fishers, create_dataset
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.train import inference
from src.model_merging.model import clone_model
import pickle
import os
import matplotlib.pyplot as plt
import omegaconf


# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def generate_plots(cfg, directory):
    models = load_models(cfg)
    fishers = load_fishers(cfg)
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

    with open(path_inference, 'rb') as f:
        inference_loss = pickle.load(f)

    with open(path_permutation, 'rb') as f:
        perm_losses = pickle.load(f)

    plt.subplot(2,1,1)
    plt.plot(perm_losses)
    plt.xlabel('Steps')
    plt.ylabel('Permutation loss')
    plt.subplot(2,1,2)
    plt.plot(inference_loss)
    plt.xlabel('Steps')
    plt.ylabel('Test loss')
    plt.axhline(avg_loss_fisher, color='b', linestyle='--')
    plt.text(0.05, avg_loss_fisher + 0.1, f'Fisher loss', color='b', fontsize=10)
    plt.axhline(avg_loss_isotropic, color='r', linestyle='--')
    plt.text(0.05, avg_loss_isotropic + 0.1, f'Isotropic loss', color='r', fontsize=10)
    plt.show()
    plt.savefig('{}plot.png'.format(directory))


if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load('./configurations/perm_mnist.yaml')
    directory = "./images/MNIST_4models_isotropic_1000epochs_m{}_seed40/"

    generate_plots(cfg, directory)