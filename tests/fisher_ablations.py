# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CogSys Section  ---  (pgare@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import tqdm
import torch.nn as nn
import math
import hydra
import random
from model_merging.data import load_grads
from model_merging.evaluation import evaluate_metamodel
from model_merging.permutation import implement_permutation, implement_permutation_grad
from model_merging.permutation import scaling_permutation, l2_permutation
import numpy as np
from torch.autograd import grad

from src.model_merging.model import clone_model
from src.model_merging.data import load_models_regression, load_fishers, create_dataset, load_grads
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.model_merging.model import MLP_regression
from src.train_regression import train, inference
from src.merge_permutation import merging_models_permutation, merging_models_weight_permutation
from model_merging.curvature_regression import compute_and_store_fisher_diagonals, compute_and_store_gradients
from model_merging.permutation import compute_permutations_init, l2_permutation


def train_models(cfg): 
    n_models = 3
    names = []
    models = []
    
    dataset = create_dataset(cfg)
    train_loader, test_loader = dataset.create_dataloaders()

    for i in range(n_models):
        model = MLP_regression(cfg)
        optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        criterion = torch.nn.MSELoss()
        name = "{}regressor_{}.pt".format(
            cfg.data.model_path,
            i
        )

        cfg.train.lr = random.randint(1,3)*cfg.train.lr
        cfg.train.epochs = random.randint(cfg.train.epochs,cfg.train.epochs+30)

        name = train(cfg, name, train_loader, test_loader, model, optimizer, criterion)
        compute_and_store_gradients(cfg, name, train_loader)
        model.eval()
        models.append(model)
        names.append(name)

        return names, models, train_loader


def ablations_fisher_regression(cfg, names, models, data_loader):
    grads = load_grads(cfg, names)
    grad = grads[0]
    model = models[0]


    # # Compute the Cholesky decomposition of the FIM
    # L = torch.cholesky(P_mm, upper=False)

    # # Solve the linear system Lz = b
    # b = torch.ones(P_mm.shape[0]) # or torch.zeros(4) depending on the dimension of the FIM
    # z = torch.triangular_solve(b.unsqueeze(1), L, lower=True)[0].squeeze()

    # # Solve the linear system L^Tx = z to obtain the inverse of the FIM
    # x = torch.triangular_solve(z.unsqueeze(1), L, upper=True)[0].squeeze()
    # iP_mm = torch.mm(x.unsqueeze(1), x.unsqueeze(0))
    
    params = model.get_trainable_parameters()
    grad =  nn.utils.parameters_to_vector(grad)
    perm = torch.randperm(len(grad))
    m = 200

    grads_r = grad[perm[m:]]
    grads_m = grad[perm[:m]]

    P_mr = torch.outer(grads_m, grads_r) / 10000 
    P_mm = torch.outer(grads_m, grads_m) / 10000 

    eigvals = np.linalg.eigvalsh(P_mm)
    is_psd = np.all(eigvals >= 0)

    # print("P_mm:\n", P_mm)
    print("Eigenvalues of P_mm:", eigvals)
    print("Is P_mm positive semidefinite?", is_psd)


@hydra.main(config_path="./configurations", config_name="regression.yaml")
def ablate_the_fisher(cfg):
    names, models, data_loader = train_models(cfg)
    ablations_fisher_regression(cfg, names, models, data_loader)


if __name__ == "__main__":
    ablate_the_fisher()
