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
import math

from fisher.model_merging.model import MLP
from fisher.model_merging.data import MNIST, load_models, load_fishers, load_grads, create_dataset
from fisher.model_merging.merging import merging_models_fisher, merging_models_isotropic

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

############################################
# Evaluation
############################################

def inference(cfg, model, test_loader, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avg_loss = [0] * len(cfg.data.classes)
    count = [0] * len(cfg.data.classes)
    y_classes = dict(zip(cfg.data.classes, range(len(cfg.data.classes))))

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = model(x.to(device))
            batch_onehot = y.apply_(lambda i: y_classes[i])
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            avg_loss[y] += loss.item()
            count[y] += 1

            if batch_idx == 100 and cfg.train.plot_sample:
                probs = torch.softmax(out[0], dim=-1)
                plt.bar(np.arange(len(cfg.data.classes)), probs.cpu().numpy())
                plt.xlabel("Number of classes")
                plt.ylabel("Class probabilities for y={}".format(y))
                plt.xticks(np.arange(len(cfg.data.classes)))
                plt.show()

    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.classes))]

    print("Average loss {}".format(sum(avg_loss)/len(avg_loss)))

    if cfg.train.plot:
        plt.bar(list(y_classes.keys()), avg_loss)
        plt.xlabel("Number of classes")
        plt.ylabel("Average Test Loss")
        plt.xticks(list(y_classes.keys()))
        plt.show()

        print("")

############################################
# Permutation
###########################################

def logprob_normal(x, mu, precision):
    n = x.shape[0]    
    precision += 10e-5

    # return -0.5 * n * torch.log(2 * torch.tensor([math.pi])) - 0.5 * n * torch.log(var) - 0.5 * torch.sum((x - mu)**2 / var)
    log_p = -0.5 * n * torch.log(2 * torch.tensor([math.pi])) + 0.5 * torch.log(precision).sum() - 0.5 * torch.sum((x - mu)**2 * precision)
    print("3rd term {}".format(- 0.5 * torch.sum((x - mu)**2 * precision)))
    print("x {}".format(x))
    print("mu {}".format(mu))
    return log_p

def perm_loss(cfg, metamodel, models, grads):
    params = metamodel.get_trainable_parameters()
    metatheta = nn.utils.parameters_to_vector(params)
    prior_mean = torch.zeros(metatheta.shape[0])
    prior_cov = cfg.train.weight_decay * torch.ones(metatheta.shape[0])

    prior = -((len(models)-1)/len(models))*logprob_normal(metatheta, prior_mean, prior_cov).sum()

    n_dim = len(metatheta)
    n_perm = cfg.data.permutations
    n_models = cfg.data.n_models
    l = 0

    for m in range(1,2):
        m = 2000

        for p in range(1,n_perm):
            for k in range(n_models):
                perm = torch.randperm(n_dim)
                model = models[k]
                grad = grads[k]
                params = model.get_trainable_parameters()
                theta = nn.utils.parameters_to_vector(params)
                grad =  nn.utils.parameters_to_vector(grad)

                theta_r = theta[perm[m:]]
                theta_m = theta[perm[:m]]
                metatheta_r = metatheta[perm[m:]]
                metatheta_m = metatheta[perm[:m]]
                
                grads_r = grad[perm[m:]]
                grads_m = grad[perm[:m]]
                precision_m = grads_m ** 2
                precision_mr = torch.outer(grads_m, grads_r)

                m_pred = theta_m + 1/precision_m @ precision_mr @ (metatheta_r - theta_r)
                l1 = logprob_normal(metatheta_m, m_pred, precision_m).sum()
                print("l1 {}".format(l1))
                l += l1
    
    loss = prior + l/(n_models*n_perm*(1))
    # loss = perm_loss/(n_models*n_perm*(1))

    return -loss


def merging_models_permutation(cfg, metamodel, models, grads):
    optimizer = optim.Adam(metamodel.parameters(), lr=cfg.train.lr)#, momentum=cfg.train.momentum)
    pbar = tqdm.trange(cfg.train.epochs)

    perm_losses = []

    for it in pbar:
        optimizer.zero_grad()
        l = perm_loss(cfg, metamodel, models, grads)
        l.backward()      # Backward pass <- computes gradients
        optimizer.step()
        perm_losses.append(l.item())
        pbar.set_description(f'[Loss: {l.item():.3f}')
        break
    
    perm_losses = [x for x in perm_losses if x > 0]
    plt.plot(perm_losses)
    plt.show()
    
    return metamodel

############################################
# Main
############################################

@hydra.main(config_path="./configurations", config_name="perm.yaml")
def main(cfg):
    grads = load_grads(cfg)
    fishers = load_fishers(cfg)
    models = load_models(cfg)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = create_dataset(cfg)
    test_loader = dataset.create_inference_dataloader()

    # FISHER
    fisher_model = merging_models_fisher(cfg, models, fishers)
    inference(cfg, fisher_model, test_loader, criterion)

    # ISOTROPIC
    isotropic_model = merging_models_isotropic(cfg, models)
    inference(cfg, isotropic_model, test_loader, criterion)

    # PERMUTATION
    metamodel = MLP(cfg)

    # metamodel = isotropic_model
    perm_model = merging_models_permutation(cfg, metamodel, models, grads)
    inference(cfg, perm_model, test_loader, criterion)


if __name__=="__main__":
    main()