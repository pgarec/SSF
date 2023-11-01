# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CROMAI  ---  (pol.garcia@bsc.es)
# Barcelona Supercomputing Center (BSC)

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
from model_merging.model import get_mergeable_variables
import numpy as np
import pickle
import os
import time


def evaluate_permutation(cfg, metamodel, models, test_loader, criterion, model_names = []):
    if model_names != []:
        grads = load_grads(cfg, model_names)
    else:
        grads = load_grads(cfg)

    metamodel = merging_models_permutation(cfg, metamodel, models, grads, test_loader = "", criterion="")
    avg_loss, count = evaluate_metamodel(cfg, metamodel, criterion, test_loader)

    return avg_loss, count


def evaluate_model(model, val_loader, criterion, llm=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_loss = 0
    model = model.to(device)

    if llm:
        for input in val_loader:
            input_ids = torch.stack(input["input_ids"], dim=-1)
            attention_mask = torch.stack(input["attention_mask"], dim=-1)
            model_predictions = model(input_ids, attention_mask=attention_mask).logits
            model_predictions = torch.argmax(model_predictions, axis=-1)
            criterion.add_batch(predictions=model_predictions, references=input["label"])
        return criterion.compute()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            out = model(x.to(device))
            loss = criterion(out, y.to(device))
            avg_loss += loss

    return avg_loss / len(val_loader)    


def logprob_normal(x, mu, precision):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_p = -0.5*torch.log(2*torch.tensor([math.pi])).to(device) + 0.5*torch.log(precision) - (0.5*precision*(x - mu)**2)

    return log_p


def permutation_loss(cfg, metamodel, models, constants):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metatheta = nn.utils.parameters_to_vector(get_mergeable_variables(metamodel)).to(device)

    n_perm = cfg.data.permutations
    n_models = len(models)
    m = torch.tensor(cfg.data.m).to(device)
    maximum = torch.tensor(cfg.data.maximum).to(device)

    loss = 0.0

    cond_prior_prec = cfg.train.weight_decay * torch.ones(m).to(device)
    permuted_indices = constants["permuted_indices"]
    theta_models = constants["theta_models"]
    grad_models = constants["grad_models"]
    fisher_models = constants["fisher_models"]

    for perm in permuted_indices:
        for model in range(n_models):
            theta = theta_models[model]
            grad = grad_models[model]
            fisher = fisher_models[model]
            
            p_time = time.time()
            if maximum == -1:
                theta_r = theta[perm[m:]]
                metatheta_r = metatheta[perm[m:]]
            else:
                theta_r = theta[perm[m:maximum]]
                metatheta_r = metatheta[perm[m:maximum]]
            theta_m = theta[perm[:m]]
            metatheta_m = metatheta[perm[:m]]

            if maximum == -1:
                grads_r = grad[perm[m:]]
            else:
                grads_r = grad[perm[m:maximum]]
            grads_m = grad[perm[:m]]

            avg_grad_m = grads_m / cfg.data.n_examples
            avg_grad_r = grads_r / cfg.data.n_examples

            v = avg_grad_m
            u = fisher[perm[:m]]
            delta = u - v**2

            P_mr = torch.outer(avg_grad_m, avg_grad_r)
            P_mm = torch.outer(avg_grad_m, avg_grad_m) + torch.diag(delta) + torch.eye(m).to(device) * cfg.train.weight_decay

            # Original inversion
            m_pred = theta_m - torch.linalg.solve(P_mm, P_mr @ (metatheta_r - theta_r))
            p_pred = torch.diagonal(P_mm)
            posterior = logprob_normal(metatheta_m, m_pred, p_pred).sum()
            
            # Compute prior
            prior = -(1 - (1 / n_models)) * logprob_normal(metatheta_m, torch.zeros(m).to(device), cond_prior_prec).sum()
            loss += (posterior + prior) / (m * n_perm)
        
    return -loss


def merging_models_permutation(cfg, metamodel, models, grads, fishers, test_loader = "", llm=False, criterion="", plot=False, store=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metamodel = metamodel.to(device)
    optimizer = optim.SGD(get_mergeable_variables(metamodel), lr=cfg.train.perm_lr, momentum=cfg.train.momentum)
    pbar = tqdm.trange(cfg.train.epochs_perm)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        metamodel = nn.DataParallel(metamodel)

    perm_loss = []
    inference_loss = []
    
    start_time = time.time()

    # compute constants
    constants={}
    n_perm = cfg.data.permutations
    params_models = [get_mergeable_variables(model) for model in models]
    constants["theta_models"] = [nn.utils.parameters_to_vector(params).to(device) for params in params_models]
    constants["grad_models"] = [nn.utils.parameters_to_vector(grad).to(device) for grad in grads]
    constants["fisher_models"] = [nn.utils.parameters_to_vector(fisher).to(device) for fisher in fishers]
    n_dim = len(constants["theta_models"][0])
    constants["permuted_indices"] = [torch.randperm(n_dim).to(device) for _ in range(n_perm)]


    for it in pbar:
        p_time = time.time()
        l = permutation_loss(cfg, metamodel, models, constants)
        # print(f"Permutation loss time: {time.time() - p_time:.2f} seconds")

        b_time = time.time()
        l.backward()    
        pbar.set_description(f'[Loss: {-l.item():.3f}')  
        optimizer.step()
        # print(f"Optimization time: {time.time() - b_time:.2f} seconds")

        if it % 100 == 0:
            inf_loss = evaluate_model(metamodel, test_loader, criterion, llm)
            inference_loss.append(inf_loss)
            perm_loss.append(-l.item())
        optimizer.zero_grad()

        if it % 10000 == 0:
            if cfg.data.dataset == "PINWHEEL":
                name = "{}metamodel_{}_{}epochs_{}m_{}classes".format(cfg.data.model_path, cfg.data.dataset, it, cfg.data.m, cfg.data.n_classes)
            
            elif cfg.data.dataset == "SST2":
                name = "{}metamodel_{}_{}epochs_{}m".format(cfg.data.model_path, cfg.data.dataset, it, cfg.data.m)

            elif cfg.data.dataset == "SNELSON":
                name = "{}metamodel_{}_{}epochs_{}m".format(cfg.data.model_path, cfg.data.dataset, it, cfg.data.m)
            
            else:
                name = "{}metamodel_{}_{}models_{}epochs_{}m_{}classes".format(cfg.data.model_path, cfg.data.dataset, len(list(cfg.models)), it, cfg.data.m, cfg.data.n_classes)
            torch.save(metamodel.state_dict(), name)

    elapsed_time = time.time() - start_time
    print("Elapsed time merging {}".format(elapsed_time))

    if plot:
        if cfg.data.dataset == "PINWHEEL":
            directory = "./images/{}_{}_m{}_{}epochs_seed{}/".format(cfg.data.dataset, cfg.train.initialization, cfg.data.m, cfg.train.epochs_perm, cfg.train.torch_seed)
        
        elif cfg.data.dataset == "SST2":
            directory = "./images/{}_{}_m{}_{}epochs_seed{}/".format(cfg.data.dataset, cfg.train.initialization, cfg.data.m, cfg.train.epochs_perm, cfg.train.torch_seed)
        
        elif cfg.data.dataset == "SNELSON":
            directory = "./images/{}_{}_m{}_{}epochs_seed{}/".format(cfg.data.dataset, cfg.train.initialization, cfg.data.m, cfg.train.epochs_perm, cfg.train.torch_seed)

        else:
            directory = "./images/{}_{}models_{}_m{}_{}epochs_seed{}/".format(cfg.data.dataset, len(list(cfg.models)), cfg.train.initialization, cfg.data.m, cfg.train.epochs_perm, cfg.train.torch_seed)

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.subplot(2,1,1)
        plt.plot(perm_loss)
        plt.xlabel('Permutations')
        plt.ylabel('Permutation loss')
        plt.subplot(2,1,2)
        plt.plot(inference_loss)
        plt.xlabel('Steps')
        plt.ylabel('Test loss')
        plt.show()
        plt.savefig('{}plot.png'.format(directory))

    if store:
        # with open('{}inference_loss'.format(directory), 'wb') as f:
        #     pickle.dump(inference_loss, f)

        # with open('{}perm_loss'.format(directory), 'wb') as f:
        #     pickle.dump(perm_loss, f)
        print("Inference losses: {}".format(inference_loss))
        print("Permutation losses: {}".format(perm_loss))

    return metamodel, inference_loss, perm_loss
