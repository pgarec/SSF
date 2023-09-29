# -*- coding: utf-8 -*-
# Copyright (c) 2023 Pol Garcia Recasens
# CogSys Section  ---  (pgare@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import hydra
import torch.nn as nn
import omegaconf
import numpy as np

from src.model_merging.model import MLP, CNNMnist, clone_model
from src.model_merging.data import load_models_cnn, load_fishers, load_grads, create_dataset, load_models
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.train import inference
from src.merge_permutation import merging_models_permutation

# CONFIGURATION
cfg = omegaconf.OmegaConf.load('./configurations/perm_fmnist.yaml')
seed = cfg.train.torch_seed

if seed > -1:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_prepare_data(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grads = load_grads(cfg)
    models = load_models_cnn(cfg)
    models = [model.to(device) for model in models]
    fishers = load_fishers(cfg)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = create_dataset(cfg)
    train_loader,_ = dataset.create_dataloaders()
    test_loader = dataset.create_inference_dataloader()

    return device, grads, models, fishers, criterion, test_loader, train_loader


def print_average_loss(model_name, avg_loss):
    print("{} - Average loss {}".format(model_name, avg_loss))


def main_operations(cfg, device, grads, models, fishers, criterion, test_loader, train_loader):
    # Original Model
    original_model = models[0]
    avg_loss = inference(cfg, original_model, test_loader, criterion)
    print_average_loss("Model 0", avg_loss)

    # Random Untrained Model
    random_model = CNNMnist(cfg).to(device)
    avg_loss = inference(cfg, random_model, test_loader, criterion)
    print_average_loss("Random untrained", avg_loss)

    # Fisher Model
    output_model = clone_model(original_model, cfg)
    fisher_model = merging_models_fisher(output_model, models, fishers)
    avg_loss = inference(cfg, fisher_model, test_loader, criterion)
    print_average_loss("Fisher", avg_loss)

    # Isotropic Model
    output_model = clone_model(original_model, cfg)
    isotropic_model = merging_models_isotropic(output_model, models)
    avg_loss = inference(cfg, isotropic_model, test_loader, criterion)
    print_average_loss("Isotropic", avg_loss)

    print("m: {}".format(cfg.data.m))
    print("seed: {}".format(cfg.train.torch_seed))
    print("permutations: {}".format(cfg.data.permutations))

    # Permutation Model
    metamodel = CNNMnist(cfg)
    cfg.data.n_examples = cfg.data.grad_samples
    cfg.train.initialization = "MLP"
    perm_model, _, _ = merging_models_permutation(cfg, metamodel, models, grads, fishers, test_loader, criterion, plot=True, store=True)
    avg_loss = inference(cfg, perm_model, test_loader, criterion)
    print_average_loss("Permutation", avg_loss)


@hydra.main(config_path="./configurations", config_name="perm_mnist.yaml")
def main(cfg):
    device, grads, models, fishers, criterion, test_loader, train_loader = load_and_prepare_data(cfg)
    main_operations(cfg, device, grads, models, fishers, criterion, test_loader, train_loader)


if __name__ == "__main__":
    main()