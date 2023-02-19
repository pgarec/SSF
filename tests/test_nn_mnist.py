# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal as Normal
import hydra
import torch.optim as optim
import torch.nn.functional as F
from laplace import Laplace
from laplace.utils import ModuleNameSubnetMask
import torch.nn.functional as F

from model_merging.model import MLP 
from model_merging.data import MNIST 


palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


@hydra.main(config_path="./configurations", config_name="data.yaml")
def main(cfg):

    # DATA - Training of 2 models with different MNIST digits
    n_models = 2
    digits = [0,1]
    models = []
    posteriors = []
    criterion = torch.nn.CrossEntropyLoss()
    
    for n in range(n_models):
        cfg.data.digits = [x+n*len(digits) for x in digits]
        dataset = MNIST(cfg)
        train_loader, test_loader = dataset.create_dataloaders()
        model = MLP(cfg)
        optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        y_classes = dict(zip(cfg.data.digits, range(len(cfg.data.digits))))

        for epoch in range(cfg.train.epochs):
            model.train()
            train_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                out = model(x.to(device))
                batch_onehot = y.apply_(lambda x: y_classes[x])
                loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
                loss.backward()
                optimizer.step()
                train_loss += loss

            val_loss = 0
            with torch.no_grad():
                model.eval()
                for batch_idx, (x, y) in enumerate(test_loader):
                    out = model(x.to(device))
                    batch_onehot = y.apply_(lambda x: y_classes[x])
                    loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
                    val_loss += loss
        
        print(f"Training Loss Model {n}: {train_loss/len(train_loader):.4f}")            
        print(f"Validation Loss Model {n}: {val_loss/len(test_loader):.4f}")
        models.append(model)

        # LAPLACE APPROXIMATION   
        train_dataset = [(_,F.one_hot(torch.tensor(y_classes[x]), cfg.data.n_classes).to(torch.float)) for (_,x) in train_loader.dataset]
        test_dataset = [(_,F.one_hot(torch.tensor(y_classes[x]), cfg.data.n_classes).to(torch.float)) for (_,x) in test_loader.dataset]

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size_train,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.data.batch_size_train,
            shuffle=True,
        )

        la = Laplace(model, 'classification',
            subset_of_weights='last_layer',
            hessian_structure='diag')
        la.fit(train_loader)    
        posteriors.append(la)

    # LOSS 
    for posterior in posteriors:
        print(posterior)

        


if __name__ == "__main__": 
    main()