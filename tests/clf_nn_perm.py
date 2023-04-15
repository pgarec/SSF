# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CogSys Section  ---  (pgare@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import hydra

from src.model_merging.model import MLP
from src.model_merging.data import load_models, load_fishers, load_grads, create_dataset
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.train import inference
from src.merge_permutation import merging_models_permutation

############################################
# Main
############################################

@hydra.main(config_path="./configurations", config_name="perm.yaml")
def main(cfg):
    # if cfg.train.torch_seed > -1:
    #     torch.manual_seed(cfg.train.torch_seed)

    grads = load_grads(cfg)
    models = load_models(cfg)
    fishers = load_fishers(cfg)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = create_dataset(cfg)
    test_loader = dataset.create_inference_dataloader()

    # FISHER
    models = load_models(cfg)
    fisher_model = merging_models_fisher(cfg, models, fishers)

    # ISOTROPIC
    models = load_models(cfg)
    isotropic_model = merging_models_isotropic(cfg, models)

    # PERMUTATION
    models = load_models(cfg)
    random_model = MLP(cfg)
    metamodel = isotropic_model # siempre inicializar en isotropic -- decision que yo tomaria
    # metamodel = fisher_model
    # metamodel = MLP(cfg)

    avg_loss = inference(cfg, random_model, test_loader, criterion)
    print("Random untrained - Average loss {}".format(avg_loss))

    avg_loss = inference(cfg, isotropic_model, test_loader, criterion)
    print("Isotropic - Average loss {}".format(avg_loss))

    avg_loss = inference(cfg, fisher_model, test_loader, criterion)
    print("Fisher - Average loss {}".format(avg_loss)) 

    perm_model = merging_models_permutation(cfg, random_model, models, grads, test_loader, criterion)
    cfg.train.plot = False
    avg_loss = inference(cfg, perm_model, test_loader, criterion)
    print("Ours (after) - Average loss {}".format(avg_loss))    

if __name__=="__main__":
    main()