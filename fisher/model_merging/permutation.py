import torch
import hydra

from .model import MLP
from .data import store_file
import torch.nn as nn
import numpy as np

############################################
# Weight-space symmetry
############################################

def l2_permutation(cfg, model):
    layer_index = cfg.data.layer_weight_permutation
    parameters = {name: p for name, p in model.named_parameters()}
    weight = f"feature_map.{layer_index}.weight"
    bias = f"feature_map.{layer_index}.bias"

    assert f"feature_map.{layer_index+2}.weight" in parameters.keys()

    l2_norm = torch.linalg.norm(parameters[weight], dim=-1, ord=2)
    _, permuted_indices = torch.sort(l2_norm)
    parameters[weight] = nn.Parameter(parameters[weight][permuted_indices])
    parameters[bias] = nn.Parameter(parameters[bias][permuted_indices])

    weight_next = f"feature_map.{layer_index+2}.weight"
    parameters[weight_next] = nn.Parameter(parameters[weight_next][:, permuted_indices])

    return model


def implement_permutation(cfg, model, permuted_indices):
    layer_index = cfg.data.layer_weight_permutation
    parameters = {name: p for name, p in model.named_parameters()}
    weight = f"feature_map.{layer_index}.weight"
    bias = f"feature_map.{layer_index}.bias"

    parameters[weight] = nn.Parameter(parameters[weight][permuted_indices])
    parameters[bias] = nn.Parameter(parameters[bias][permuted_indices])

    weight_next = f"feature_map.{layer_index+2}.weight"
    parameters[weight_next] = nn.Parameter(parameters[weight_next][:, permuted_indices])

    return model


def implement_permutation_grad(cfg, grad, permuted_indices):
    layer_index = (cfg.data.layer_weight_permutation-1)*2
    grad[layer_index] = grad[layer_index][permuted_indices]
    grad[layer_index+1] = grad[layer_index+1][permuted_indices]
    grad[layer_index+2] = nn.Parameter(grad[layer_index+2][:, permuted_indices])
    
    return grad


def random_weight_permutation(model, layer_index):
    parameters = {name: p for name, p in model.named_parameters()}
    weight = f"feature_map.{layer_index}.weight"
    bias = f"feature_map.{layer_index}.bias"

    assert f"feature_map.{layer_index+2}.weight" in parameters.keys()

    num_units = parameters[weight].shape[0]
    permuted_indices = torch.randperm(num_units)

    with torch.no_grad():
        parameters[weight] = nn.Parameter(parameters[weight][permuted_indices])
        parameters[bias] = nn.Parameter(parameters[bias][permuted_indices])

        weight_next = f"feature_map.{layer_index+2}.weight"
        parameters[weight_next] = nn.Parameter(parameters[weight_next][:, permuted_indices])

    return model


def sorted_weight_permutation(model, layer_index):
    parameters = {name: p for name, p in model.named_parameters()}
    weight = f"feature_map.{layer_index}.weight"
    bias = f"feature_map.{layer_index}.bias"

    assert f"feature_map.{layer_index+2}.weight" in parameters.keys()

    _, permuted_indices = torch.sort(parameters[weight][:, 0])
    with torch.no_grad():
        parameters[weight] = nn.Parameter(parameters[weight][permuted_indices])
        parameters[bias] = nn.Parameter(parameters[bias][permuted_indices])

        weight_next = f"feature_map.{layer_index+2}.weight"
        parameters[weight_next] = nn.Parameter(parameters[weight_next][:, permuted_indices])

    return model


def indices_random_weight_permutation(model, layer_index):
    parameters = {name: p for name, p in model.named_parameters()}
    weight = f"feature_map.{layer_index}.weight"

    assert f"feature_map.{layer_index+2}.weight" in parameters.keys()

    permuted_indices = torch.randperm(parameters[weight].shape[0])

    return permuted_indices


def compute_permutations_for_model(cfg, model):    
    list_indices = []

    for p in range(cfg.data.weight_permutations):
        print(p)
        list_indices.append(indices_random_weight_permutation(model, cfg.data.layer_weight_permutation))

    return list_indices


def compute_permutations(cfg, model_name):
    model = MLP(cfg)
    model.load_state_dict(torch.load(model_name))

    print("Starting permutations computation")
    permutations = compute_permutations_for_model(cfg, model)
    print("Permutations computed. Saving to file...")
    perm_name = model_name.split('/')[-1][:-3]
    store_file(permutations, cfg.data.perm_path + perm_name)
    print("Permutations saved to file")

############################################
# Scaling symmetry
############################################

def scaling_permutation(cfg, model,layer_index=-1, scaler=-1):
    if layer_index == -1:
        layer_index = cfg.data.layer_weight_permutation
    parameters = {name: p for name, p in model.named_parameters()}
    weight = f"feature_map.{layer_index}.weight"
    bias = f"feature_map.{layer_index}.bias"

    assert f"feature_map.{layer_index+2}.weight" in parameters.keys()

    if scaler == -1:
        scaler = np.random.randint(cfg.data.scaling_perm_min, cfg.data.scaling_perm_max)

    parameters[weight] = nn.Parameter(scaler*parameters[weight])
    parameters[bias] = nn.Parameter(scaler*parameters[bias])

    weight_next = f"feature_map.{layer_index+2}.weight"
    parameters[weight_next] = nn.Parameter((1/scaler)*parameters[weight_next])

    return model

############################################
# Main
############################################

if __name__ == "__main__":
    compute_permutations()
