import os
import torch
import torch.optim as optim
import numpy as np
import hydra
import wandb
import sys
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

sys.path.append("./utils")
sys.path.append("./models")
from data_utils import get_mnist_loaders
from network import Model


@hydra.main(config_path="./configurations", config_name="basic.yaml")
def train(cfg):
    # wandb.init(project="test-project", entity="model-driven-models")
    # wandb.config = cfg
    data_path = cfg.data.data_path
    dataset = cfg.data.dataset
    n_epochs = cfg.hyperparameters.n_epochs

    if dataset == "MNIST":
        train_loader, val_loader, test_loader = get_mnist_loaders(data_path)
        assert(cfg.data.image_shape == 784)
    
    model = Model(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr, weight_decay=cfg.hyperparameters.w_dc)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.hyperparameters.step_size, gamma=cfg.hyperparameters.gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CrossEntropyLoss()

    model.cuda()
    model.train()

    for epoch in trange(n_epochs):
        training_loss = 0
        num_batches = train_loader.n_batches
        num_data = train_loader.dataset_len

        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x.to(device))
            y = torch.nn.functional.one_hot(y, cfg.data.n_classes)

            loss = criterion(out, y.float().to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f" Training loss:{loss/len(train_loader):.4f}")
            # wandb.log({"Training loss": loss/len(train_loader)})

        with torch.no_grad():
            loss_val = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                out = model(x.to(device))
                y = torch.nn.functional.one_hot(y, cfg.data.n_classes)
                loss = criterion(out, y.float().to(device))
                loss_val += loss
            
            print(f" Validation loss:{loss/len(train_loader):.4f}")
            # wandb.log({"Validation loss": loss/len(train_loader)})
        
        torch.save(model.state_dict(), "./baselines/nn/models/model_epoch{}.mod".format(epoch))
    

if __name__ == "__main__":
    train()
