# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CROMAI  ---  (pol.garcia@bsc.es)
# Barcelona Supercomputing Center (BSC)

import torch
import hydra
from model_merging.data import MNIST, load_models
from model_merging.merging import merging_models_isotropic
from model_merging.evaluation import evaluate_metamodel, evaluate_minimodels, plot


def evaluate_isotropic(cfg, models, test_loader, criterion):
    merged_model = merging_models_isotropic(cfg, models)
    avg_loss, count = evaluate_metamodel(cfg, merged_model, criterion, test_loader)

    return avg_loss, count


@hydra.main(config_path="./configurations", config_name="merge.yaml")
def isotropic(cfg):
    models = load_models(cfg)
    dataset = MNIST(cfg)
    cfg.data.batch_size_test = 1
    _, test_loader = dataset.create_dataloaders()
    criterion = torch.nn.CrossEntropyLoss()

    merged_model = merging_models_isotropic(cfg, models)
    avg_loss, _ = evaluate_metamodel(cfg, merged_model, criterion, test_loader)
    avg_loss_models, count = evaluate_minimodels(cfg, models, criterion, test_loader)

    if cfg.data.plot:
        plot(cfg, avg_loss, avg_loss_models, count, models)

    return avg_loss


if __name__ == "__main__":
    isotropic()
