# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CogSys Section  ---  (pgare@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import hydra

from fisher.model_merging.model import MLP
from fisher.model_merging.data import load_permutations, load_models, load_fishers, load_grads, create_dataset
from fisher.model_merging.merging import merging_models_fisher, merging_models_isotropic
from fisher.train import inference
from fisher.model_merging.model import clone_model
from fisher.merge_permutation import merging_models_permutation, merging_models_weight_permutation
from fisher.model_merging.permutation import compute_permutations, sorted_weight_permutation
import torch.nn as nn

############################################
# Main
############################################

@hydra.main(config_path="./configurations", config_name="perm.yaml")
def main(cfg):
    grads = load_grads(cfg)
    models = load_models(cfg)
    fishers = load_fishers(cfg)
    permutations = load_permutations(cfg)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = create_dataset(cfg)
    test_loader = dataset.create_inference_dataloader()

    params = models[0].get_trainable_parameters()
    metatheta = nn.utils.parameters_to_vector(params)
    print(len(metatheta))
    avg_loss = inference(cfg, models[0], test_loader, criterion)
    print("Model 0 - Average loss {}".format(avg_loss))

    random_model = MLP(cfg)
    avg_loss = inference(cfg, random_model, test_loader, criterion)
    print("Random untrained - Average loss {}".format(avg_loss))

    # FISHER
    models = load_models(cfg)
    output_model = clone_model(models[0], cfg)
    fisher_model = merging_models_fisher(output_model, models, fishers)

    avg_loss = inference(cfg, fisher_model, test_loader, criterion)
    print("Fisher - Average loss {}".format(avg_loss)) 

    # ISOTROPIC
    models = load_models(cfg)
    output_model = clone_model(models[0], cfg)
    isotropic_model = merging_models_isotropic(output_model, models)

    avg_loss = inference(cfg, isotropic_model, test_loader, criterion)
    print("Isotropic - Average loss {}".format(avg_loss))

    # PERMUTATION
    models = load_models(cfg)
    random_model = MLP(cfg)
    metamodel = isotropic_model # siempre inicializar en isotropic -- decision que yo tomaria
    # metamodel = fisher_model
    # metamodel = MLP(cfg)
    perm_model = merging_models_permutation(cfg, metamodel, models, grads, test_loader, criterion, plot=True)

    avg_loss = inference(cfg, perm_model, test_loader, criterion)
    print("Permutation - Average loss {}".format(avg_loss))  

    # WEIGHT PERMUTATION
    # models = load_models(cfg)
    # random_model = MLP(cfg)
    # metamodel = isotropic_model # siempre inicializar en isotropic -- decision que yo tomaria
    # # metamodel = fisher_model
    # # metamodel = MLP(cfg)
    # weight_perm_model = merging_models_weight_permutation(cfg, metamodel, models, permutations, grads, test_loader, criterion)

    # avg_loss = inference(cfg, weight_perm_model, test_loader, criterion)
    # print("Weight permutation - Average loss {}".format(avg_loss))   

if __name__=="__main__":
    main()