import os 
import torch
import numpy as np 
import hydra
import wandb
import sys

sys.path.append("./utils")
from data_utils import get_mnist_loaders


@hydra.main(config_path="./configurations", config_name="basic.yaml")
def train(cfg):
    # wandb.init(project="test-project", entity="model-driven-models")
    # wandb.config = cfg
    data_path = cfg.hyperparameters.data_path
    digits = cfg.hyperparameters.digits
    dataset = cfg.hyperparameters.dataset

    if dataset == 'MNIST':
        train_loader, val_loader, test_loader = get_mnist_loaders(data_path, digits)


if __name__ == "__main__":
    train()