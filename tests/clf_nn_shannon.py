# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CogSys Section  ---  (pgare@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import matplotlib.pyplot as plt
import hydra
import torch.optim as optim
import tqdm

from fisher.model_merging.merging import merging_models_isotropic 
from fisher.model_merging.data import load_models, load_fishers, create_dataset
from fisher.train import inference
from fisher.merge_shannon import shannon_loss

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