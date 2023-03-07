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

from fisher.model_merging.merging import merging_models_isotropic 
from fisher.model_merging.data import load_models, load_fishers, create_dataset

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
    var += 0.000001
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
# Evalutation
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

    print(avg_loss)
    print("Average loss shannon loss {}".format(sum(avg_loss)/len(avg_loss)))

    if cfg.train.plot:
        plt.bar(list(y_classes.keys()), avg_loss)
        plt.xlabel("Number of classes")
        plt.ylabel("Average Test Loss")
        plt.xticks(list(y_classes.keys()))
        plt.show()

        print("")

############################################
# Main
############################################

@hydra.main(config_path="./configurations", config_name="data.yaml")
def main(cfg):
    fishers = load_fishers(cfg)
    models = load_models(cfg)
    # metamodel = MLP(cfg)
    # metamodel = clone_model(models[0], cfg)
    metamodel = merging_models_isotropic(cfg, models)
    optimizer = optim.SGD(metamodel.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)
    pbar = tqdm.trange(cfg.train.epochs)

    shannon_losses = []

    for it in pbar:
        optimizer.zero_grad()
        l = shannon_loss(metamodel, models, fishers)
        l.backward()      # Backward pass <- computes gradients
        optimizer.step()
        shannon_losses.append(l.item())
        pbar.set_description(f'[Loss: {l.item():.3f}')

    criterion = torch.nn.CrossEntropyLoss()
    dataset = create_dataset(cfg)
    test_loader = dataset.create_inference_dataloader()
    inference(cfg, models[0], test_loader, criterion)

    plt.plot(shannon_losses)
    plt.show()
    

if __name__ == "__main__": 
    main()