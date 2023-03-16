import torch
import hydra

from .model import MLP
from .data import store_file
import torch.nn as nn


def weight_permutation(model, layer_index):
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


def compute_permutations_for_model(model):
    params = model.get_trainable_parameters()
    theta = nn.utils.parameters_to_vector(params)
    

def compute_permutations(cfg, model_name):
    model = MLP(cfg)
    model.load_state_dict(torch.load(model_name))

    print("Starting permutations computation")
    permutations = compute_permutations_for_model(model)
    print("Permutations computed. Saving to file...")
    perm_name = model_name.split('/')[-1][:-3]
    store_file(permutations, cfg.data.perm_path + perm_name)
    print("Fisher saved to file")


if __name__ == "__main__":
    compute_permutations()
