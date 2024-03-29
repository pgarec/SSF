# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CROMAI  ---  (pol.garcia@bsc.es)
# Barcelona Supercomputing Center (BSC)

import torch
import torch.nn.functional as F
import torch.optim as optim
import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf

from model_merging.model import MLP, CNNMnist
from model_merging.curvature import compute_and_store_fisher_diagonals, compute_and_store_gradients
from model_merging.data import create_dataset
from model_merging.permutation import l2_permutation, compute_permutations

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

# CONFIGURATION
cfg = omegaconf.OmegaConf.load('./configurations/train.yaml')
seed = cfg.train.torch_seed

if seed > -1:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg, name, train_loader, test_loader, model, optimizer, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    y_classes = dict(zip(cfg.data.classes, range(len(cfg.data.classes))))
    for epoch in range(cfg.train.epochs):
        train_loss = 0
        for _, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x.to(device))
            batch_onehot = y.apply_(lambda x: y_classes[x]).to(device)
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            loss.backward()
            optimizer.step()           
            train_loss += loss
            # wandb.log({"Training loss": loss/len(train_loader)})

        print(
            f"Epoch [{epoch + 1}/{cfg.train.epochs}], Training Loss: {train_loss/len(train_loader):.4f}"
        )

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for _, (x, y) in enumerate(test_loader):
                out = model(x.to(device))
                batch_onehot = y.apply_(lambda x: y_classes[x]).to(device)
                loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
                val_loss += loss
            # wandb.log({"Validation loss": loss/len(val_loader)})
            print(
                f"Epoch [{epoch + 1}/{cfg.train.epochs}], Validation Loss: {val_loss/len(test_loader):.4f}"
            )

    print("")
    torch.save(model.state_dict(), name)
    print(name)


def evalute_normalize(cfg, model, test_loader, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avg_loss = [0] * len(cfg.data.classes)
    count = [0] * len(cfg.data.classes)
    y_classes = dict(zip(cfg.data.classes, range(len(cfg.data.classes))))

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = model(x.to(device))
            batch_onehot = y.apply_(lambda i: y_classes[i]).to(device)
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            avg_loss[y] += loss.item()
            count[y] += 1 #x.shape[0]

            if batch_idx == 0 and cfg.train.plot_sample:
                probs = torch.softmax(out[0], dim=-1)
                plt.bar(np.arange(len(cfg.data.classes)), probs.cpu().numpy())
                plt.xlabel("Number of classes")
                plt.ylabel("Class probabilities for y={}".format(y))
                plt.xticks(np.arange(len(cfg.data.classes)))
                plt.show()
            
    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.classes))]

    if cfg.train.plot:
        plt.bar(list(y_classes.keys()), avg_loss)
        plt.xlabel("Number of classes")
        plt.ylabel("Average Test Loss")
        plt.xticks(list(y_classes.keys()))
        plt.show()

        print("")

    return sum(avg_loss) / len(avg_loss)


@hydra.main(config_path="./configurations", config_name="train.yaml")
def main(cfg):

    seed = cfg.train.torch_seed
    if seed > -1:
        np.random.seed(seed)
        torch.manual_seed(seed)

    else:
        dataset = create_dataset(cfg)   
        train_loader, test_loader = dataset.create_dataloaders(unbalanced=cfg.data.unbalanced)
        model = CNNMnist(cfg)
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        if cfg.data.unbalanced == []:
            name = "{}{}_{}_epoch{}.pt".format(
                cfg.data.model_path,
                cfg.data.dataset,
                "".join(map(str, cfg.data.classes)), cfg.train.epochs
            )
        else:
            name = "{}{}_{}_epoch{}_{}.pt".format(
                cfg.data.model_path,
                cfg.data.dataset,
                "".join(map(str, cfg.data.classes)), cfg.train.epochs,
                "".join(map(str, cfg.data.unbalanced)))

        train(cfg, name, train_loader, test_loader, model, optimizer, criterion)
        test_loader = dataset.create_inference_dataloader()
        evalute_normalize(cfg, model, test_loader, criterion)

        if cfg.train.fisher_diagonal:
            compute_and_store_fisher_diagonals(model, name, cfg.data.fisher_path, test_loader)

        if cfg.train.fisher_gradients:
            compute_and_store_gradients(model, name, cfg.data.grad_path, test_loader)


if __name__ == "__main__":
    main()
