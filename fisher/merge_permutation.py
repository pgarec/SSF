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
from model_merging.data import load_grads
from model_merging.evaluation import evaluate_metamodel


def weight_permutation(model, layer_index):
    parameters = {name: p for name, p in model.named_parameters()}
    weight = f"feature_map.{layer_index}.weight"
    bias = f"feature_map.{layer_index}.bias"

    assert f"feature_map.{layer_index+2}.weight" in parameters.keys()

    num_units = parameters[weight].shape[0]
    permuted_indices = torch.randperm(num_units)

    with torch.no_grad():
        parameters[weight] = nn.Parameter(parameters[weight][permuted_indices])
        parameters[bias] = nn.Parameter(parameters[bias][permuted_indices])

        weight_next = f"feature_map.{layer_index+2}.weight"
        parameters[weight_next] = nn.Parameter(parameters[weight_next][:, permuted_indices])

    return model


def logprob_normal(x, mu, precision):
    n = x.shape[0]    
    precision = precision + 10e-5

    # return -0.5 * n * torch.log(2 * torch.tensor([math.pi])) - 0.5 * n * torch.log(var) - 0.5 * torch.sum((x - mu)**2 / var)
    # log_p = -0.5 * n * torch.log(2 * torch.tensor([math.pi])) + 0.5 * torch.log(precision).sum() - 0.5 * torch.sum((x - mu)**2 * precision)

    log_p = -0.5*torch.log(2*torch.tensor([math.pi])) + 0.5*torch.log(precision) - 0.5*precision*(x - mu)**2

    return log_p


def perm_loss(cfg, metamodel, models, grads):
    params = metamodel.get_trainable_parameters()
    metatheta = nn.utils.parameters_to_vector(params)

    n_dim = len(metatheta)
    n_perm = cfg.data.permutations
    n_models = len(models)

    loss = 0.0
    m = 10
    for p in range(n_perm):
        perm = torch.randperm(n_dim)
        # k = torch.randperm(n_models)[0]
        for k in range(n_models):

            model = models[k]
            grad = grads[k]
            params = model.get_trainable_parameters()
            theta = nn.utils.parameters_to_vector(params)
            grad =  nn.utils.parameters_to_vector(grad)

            theta_r = theta[perm[m:]]
            theta_m = theta[perm[:m]]
            metatheta_r = metatheta[perm[m:]].detach()
            metatheta_m = metatheta[perm[:m]]
            
            grads_r = grad[perm[m:]]
            grads_m = grad[perm[:m]]
            precision_m = torch.clamp(grads_m ** 2, min=1e-20)
            precision_mr = torch.outer(grads_m, grads_r)

            # m_pred = theta_m - (1/precision_m) @ (precision_mr @ (metatheta_r - theta_r))
            m_pred = theta_m - (1/precision_m) * (precision_mr @ (metatheta_r - theta_r))
            posterior = n_models*logprob_normal(metatheta_m, m_pred, precision_m).sum()
            posterior = logprob_normal(metatheta_m, m_pred, precision_m).sum()

            cond_prior_m = torch.zeros(m)    
            cond_prior_prec = cfg.train.weight_decay * torch.ones(m)
            prior = -(1 - (1/n_models))*logprob_normal(metatheta_m, cond_prior_m, cond_prior_prec).sum()

            loss += (posterior + prior)/(m * n_perm)

    return -loss


def merging_models_permutation(cfg, metamodel, models, grads, test_loader = "", criterion=""):
    # optimizer = optim.Adam(metamodel.parameters(), lr=cfg.train.lr)
    # optimizer = optim.SGD(metamodel.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)#, momentum=cfg.train.momentum)
    optimizer = optim.SGD(metamodel.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)
    pbar = tqdm.trange(cfg.train.epochs)
    cfg.data.plot = False

    perm_losses = []
    inference_loss = []

    for it in pbar:
        optimizer.zero_grad()
        l = perm_loss(cfg, metamodel, models, grads)
        l.backward()      # Backward pass <- computes gradients
        optimizer.step()
        perm_losses.append(-l.item())
        pbar.set_description(f'[Loss: {-l.item():.3f}')
        # if it % 100:
            # pass
            # inference_loss.append(inference(cfg, metamodel, test_loader, criterion))

    # perm_losses = [x for x in perm_losses if x > 0]
    if cfg.data.plot:
        # plt.subplot(2,1,1)
        plt.plot(perm_losses)
        plt.xlabel('Permutations')
        plt.ylabel('Loss')
        # plt.subplot(2,1,2)
        # plt.plot(inference_loss)
        plt.show()
    
    return metamodel


def evaluate_permutation(cfg, metamodel, models, test_loader, criterion, model_names = []):
    if model_names != []:
        grads = load_grads(cfg, model_names)
    else:
        grads = load_grads(cfg)

    metamodel = merging_models_permutation(cfg, metamodel, models, grads, test_loader = "", criterion="")
    avg_loss, count = evaluate_metamodel(cfg, metamodel, criterion, test_loader)

    return avg_loss, count