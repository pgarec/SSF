import os 
import torch
import numpy as np 
import hydra
import wandb
import sys

sys.path.append("./utils")
from dataset import MNISTDataset


@hydra.main(config_name="configurations/config.yaml")
def train(cfg):
    wandb.init(project="test-project", entity="model-driven-models")
    wandb.config = cfg
    dataset = MNISTDataset('../dataset/mnist.npz', digits=[0,1,2])
    train()


if __name__ == "__main__":
    train()