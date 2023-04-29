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
import random
from model_merging.data import load_grads
from model_merging.evaluation import evaluate_metamodel
from model_merging.permutation import implement_permutation, implement_permutation_grad
from model_merging.permutation import scaling_permutation, l2_permutation
import numpy as np
import pickle
import os


def logprob_normal(x, mu, precision):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = x.shape[0]    
    precision = precision # + 10e-5
    
    log_p = -0.5*torch.log(2*torch.tensor([math.pi])).to(device) + 0.5*torch.log(precision) - 0.5*precision*(x - mu)**2

    return log_p


def evaluate_model(model, val_loader, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_loss = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            out = model(x.to(device))
            loss = criterion(out, y.to(device))
            avg_loss += loss

    return avg_loss / len(val_loader)


def perm_loss_fisher(cfg, metamodel, models, grads):
    params = metamodel.get_trainable_parameters()
    metatheta = nn.utils.parameters_to_vector(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_dim = len(metatheta)
    n_perm = cfg.data.permutations
    n_models = len(models)

    loss = 0.0
    m = cfg.data.m
    for p in range(n_perm):
        perm = torch.randperm(n_dim).to(device)
        perm.requires_grad = False
        for k in range(n_models):
            model = models[k].to(device)
            grad = grads[k]
            params = model.get_trainable_parameters()
            theta = nn.utils.parameters_to_vector(params)
            grad =  nn.utils.parameters_to_vector(grad)

            theta_r = theta[perm[m:]]
            theta_m = theta[perm[:m]]
            metatheta_r = metatheta[perm[m:]]# .detach()
            metatheta_m = metatheta[perm[:m]]

            grads_r = grad[perm[m:]]
            grads_m = grad[perm[:m]]

            P_mr = torch.outer(grads_m, grads_r).to(device) / cfg.data.n_examples # + cfg.train.weight_decay * torch.eye(m)
            P_mm = torch.outer(grads_m, grads_m).to(device) / cfg.data.n_examples + cfg.train.weight_decay * torch.eye(m).to(device)
            
            m_pred = theta_m - torch.linalg.solve(P_mm, P_mr) @ (metatheta_r - theta_r)
            p_pred = torch.diagonal(P_mm)
            posterior = logprob_normal(metatheta_m, m_pred, p_pred).sum()

            cond_prior_m = torch.zeros(m).to(device)    
            cond_prior_prec = cfg.train.weight_decay * torch.ones(m).to(device)
            prior = -(1 - (1/n_models))*logprob_normal(metatheta_m, cond_prior_m, cond_prior_prec).sum()
            
            loss += (posterior + prior)/(m * n_perm)

    return -loss


def merging_models_permutation(cfg, metamodel, models, grads, test_loader = "", criterion="", plot=False):
    optimizer = optim.SGD(metamodel.get_trainable_parameters(), lr=cfg.train.perm_lr, momentum=cfg.train.momentum)
    pbar = tqdm.trange(cfg.train.epochs_perm)

    perm_losses = []
    inference_loss = []

    for it in pbar:
        if it % 10:
            inf_loss = evaluate_model(metamodel, test_loader, criterion)
            inference_loss.append(inf_loss)
        optimizer.zero_grad()
        
        l = perm_loss_fisher(cfg, metamodel, models, grads)
        l.backward()      
        optimizer.step()
        perm_losses.append(-l.item())
        pbar.set_description(f'[Loss: {-l.item():.3f}')

        if it % 10000:
            if cfg.data.dataset == "PINWHEEL":
                name = "{}metamodel_{}_{}epochs_{}m_{}classes".format(cfg.data.model_path, cfg.data.dataset, it, cfg.data.m, cfg.data.n_classes)

            elif cfg.data.dataset == "SNELSON":
                name = "{}metamodel_{}_{}epochs_{}m".format(cfg.data.model_path, cfg.data.dataset, it, cfg.data.m)
            
            else:
                name = "{}metamodel_{}_{}models_{}epochs_{}m_{}classes".format(cfg.data.model_path, cfg.data.dataset, len(list(cfg.models)), it, cfg.data.m, cfg.data.n_classes)
            torch.save(metamodel.state_dict(), name)

    if plot:
        if cfg.data.dataset == "PINWHEEL":
            directory = "./images/{}_{}_m{}_{}epochs_seed{}/".format(cfg.data.dataset, cfg.train.initialization, cfg.data.m, cfg.train.epochs_perm, cfg.train.torch_seed)
        
        elif cfg.data.dataset == "SNELSON":
            directory = "./images/{}_{}_m{}_{}epochs_seed{}/".format(cfg.data.dataset, cfg.train.initialization, cfg.data.m, cfg.train.epochs_perm, cfg.train.torch_seed)

        else:
            directory = "./images/{}_{}models_{}_m{}_{}epochs_seed{}/".format(cfg.data.dataset, len(list(cfg.models)), cfg.train.initialization, cfg.data.m, cfg.train.epochs_perm, cfg.train.torch_seed)

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.subplot(2,1,1)
        plt.plot(perm_losses)
        plt.xlabel('Permutations')
        plt.ylabel('Permutation loss')
        plt.subplot(2,1,2)
        plt.plot(inference_loss)
        plt.xlabel('Steps')
        plt.ylabel('Test loss')
        plt.show()
        plt.savefig('{}plot.png'.format(directory))

        with open('{}inference_loss'.format(directory), 'wb') as f:
            pickle.dump(inference_loss, f)

        with open('{}perm_loss'.format(directory), 'wb') as f:
            pickle.dump(perm_losses, f)

    return metamodel


def scaling_perm_loss(cfg, metamodel, models, grads):
    params = metamodel.get_trainable_parameters()
    metatheta = nn.utils.parameters_to_vector(params)

    n_dim = len(metatheta)
    n_perm = cfg.data.permutations
    n_models = len(models)
    n_scalations = cfg.data.number_scalations

    loss = 0.0
    m = cfg.data.m
    for p in range(n_perm):
        perm = torch.randperm(n_dim)
        for k in range(n_models):
            model = models[k]
            grad = grads[k]
            grad =  nn.utils.parameters_to_vector(grad)
            grads_r = grad[perm[m:]] 
            grads_m = grad[perm[:m]] 

            P_mr = torch.outer(grads_m, grads_r) / 350
            P_mm = torch.outer(grads_m, grads_m) / 350 + cfg.train.weight_decay * torch.eye(m)

            # model = l2_permutation(cfg, model)

            for n in range(n_scalations):
                model = scaling_permutation(cfg, model, layer_index=cfg.data.layer_weight_permutation) 
                params = model.get_trainable_parameters()
                theta = nn.utils.parameters_to_vector(params)

                theta_r = theta[perm[m:]]
                theta_m = theta[perm[:m]]
                metatheta_r = metatheta[perm[m:]]
                metatheta_m = metatheta[perm[:m]]
                
                m_pred = theta_m - torch.linalg.solve(P_mm, P_mr) @ (metatheta_r - theta_r)
                p_pred = torch.diagonal(P_mm)
                posterior = logprob_normal(metatheta_m, m_pred, p_pred).sum()
    
                cond_prior_m = torch.zeros(m)    
                cond_prior_prec = cfg.train.weight_decay * torch.ones(m)
                prior = -(1 - (1/n_models))*logprob_normal(metatheta_m, cond_prior_m, cond_prior_prec).sum()

                loss += (posterior + prior)/(n_perm * n_models * n_scalations * m)

    return -loss


def merging_models_scaling_permutation(cfg, metamodel, models, grads, test_loader = "", criterion="", plot=False):
    optimizer = optim.SGD(metamodel.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)
    pbar = tqdm.trange(cfg.train.epochs_perm)

    perm_losses = []
    inference_loss = []

    for it in pbar:
        optimizer.zero_grad()
        l = scaling_perm_loss(cfg, metamodel, models, grads)
        l.backward()      
        optimizer.step()
        perm_losses.append(-l.item())
        pbar.set_description(f'[Loss: {-l.item():.3f}')
        if it % 10:
            inference_loss.append(evaluate_model(metamodel, test_loader, criterion))
    
    if plot:
        plt.subplot(2,1,1)
        plt.plot(perm_losses)
        plt.xlabel('Steps')
        plt.ylabel('Scaling permutation loss')
        plt.subplot(2,1,2)
        plt.plot(inference_loss)
        plt.xlabel('Permutations')
        plt.ylabel('Test loss')
        plt.show()
    
    return metamodel


def weight_perm_loss(cfg, metamodel, models, permutations, grads):
    params = metamodel.get_trainable_parameters()
    metatheta = nn.utils.parameters_to_vector(params)

    n_dim = len(metatheta)
    n_perm = cfg.data.permutations
    n_models = len(models)

    loss = 0.0
    m = cfg.data.m
    for _ in range(n_perm):
        perm = torch.randperm(n_dim)
        
        for k in range(n_models):
            model = models[k]
            perms = permutations[k]

            for perm_model in perms:
                grad = grads[k]
                model = implement_permutation(model, perm_model, cfg.data.layer_weight_permutation)            
                grad = implement_permutation_grad(grad, perm_model, cfg.data.layer_weight_permutation)
                params = model.get_trainable_parameters()
                theta = nn.utils.parameters_to_vector(params)

                grad =  nn.utils.parameters_to_vector(grad)
                theta_r = theta[perm[m:]]
                theta_m = theta[perm[:m]]
                metatheta_r = metatheta[perm[m:]]
                metatheta_m = metatheta[perm[:m]]

                grads_r = grad[perm[m:]]
                grads_m = grad[perm[:m]]
                P_mr = torch.outer(grads_m, grads_r) / cfg.data.n_examples
                P_mm = torch.outer(grads_m, grads_m) / cfg.data.n_examples + cfg.train.weight_decay * torch.eye(m)
                
                m_pred = theta_m - torch.linalg.solve(P_mm, P_mr) @ (metatheta_r - theta_r)
                p_pred = torch.diagonal(P_mm)
                posterior = logprob_normal(metatheta_m, m_pred, p_pred).sum()

                cond_prior_m = torch.zeros(m)    
                cond_prior_prec = cfg.train.weight_decay * torch.ones(m)
                prior = -(1 - (1/n_models))*logprob_normal(metatheta_m, cond_prior_m, cond_prior_prec).sum()

                loss += (posterior + prior)/(m * n_perm * len(permutations[k]))

    return -loss


def merging_models_weight_permutation(cfg, metamodel, models, permutations, grads, test_loader = "", criterion="", plot=False):
    optimizer = optim.SGD(metamodel.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)#, momentum=cfg.train.momentum)
    pbar = tqdm.trange(cfg.train.epochs_perm)

    perm_losses = []
    inference_loss = []

    for it in pbar:
        optimizer.zero_grad()
        l = weight_perm_loss(cfg, metamodel, models, permutations, grads)
        l.backward()      # Backward pass <- computes gradients
        optimizer.step()
        perm_losses.append(-l.item())
        pbar.set_description(f'[Loss: {-l.item():.3f}')

        if it % 10:
            inference_loss.append(evaluate_model(metamodel, test_loader, criterion))

    if plot:
        plt.subplot(2,1,1)
        plt.plot(perm_losses)
        plt.xlabel('Permutations')
        plt.ylabel('Permutation loss')
        plt.subplot(2,1,2)
        plt.plot(inference_loss)
        plt.xlabel('Permutations')
        plt.ylabel('Test loss')
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