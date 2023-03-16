import torch
import torch.nn.functional as F
import torch.optim as optim
import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb

from fisher.model_merging.model import MLP
from fisher.model_merging.fisher import compute_fisher_diags, compute_fisher_grads
from fisher.model_merging.data import create_dataset
from fisher.train import train, inference 
from fisher.evaluation import evaluate_techniques


@hydra.main(config_path="./configurations", config_name="multitrain.yaml")
def main(cfg):
    name_models = []
    dataset = create_dataset(cfg)
    train_loader, test_loader = dataset.create_dataloaders(unbalanced=cfg.data.unbalanced)
    test_loader = dataset.create_inference_dataloader()
  
    for r in range(cfg.train.n_models):
        cfg.train.lr = np.random.uniform(0.001, 0.1)
        model = MLP(cfg)
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        path = "{}model_{}.pt".format(
                cfg.data.model_path,
                r)

        train(cfg, path, train_loader, test_loader, model, optimizer, criterion)
        name_models.append("model_{}.pt".format(r))
        inference(cfg, model, test_loader, criterion)

        if cfg.train.fisher_diagonal:
            compute_fisher_diags(cfg, path)

        if cfg.train.fisher_gradients:
            compute_fisher_grads(cfg, path)

    evaluate_techniques(cfg, name_models) 


if __name__ == "__main__":
    main()
