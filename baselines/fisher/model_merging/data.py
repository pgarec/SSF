import torch
import pickle
from model_merging.model import MLP
from model_merging.datasets.mnist import MNIST
from model_merging.datasets.fmnist import FashionMNIST
from model_merging.datasets.cifar import CIFAR10, CIFAR100


def load_models(cfg):
    models = []

    for model_name in cfg.models:
        model = MLP(cfg)
        model.load_state_dict(torch.load(cfg.data.model_path+cfg.models[model_name]+".pt"))
        models.append(model)

    return models


def store_file(file, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(file, f)


def load_file(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def load_fishers(cfg):
    fishers = []

    for model_name in cfg.models:
        path = cfg.data.fisher_path + cfg.models[model_name]
        fisher = load_file(path)
        fishers.append(fisher)

    return fishers


def create_dataset(cfg):
    if cfg.data.dataset == "MNIST":
        dataset = MNIST(cfg)

    elif cfg.data.dataset == "FashionMNIST":  
        dataset = FashionMNIST(cfg)
    
    elif cfg.data.dataset == "CIFAR10":
        dataset = CIFAR10(cfg)
    
    elif cfg.data.dataset == "CIFAR100":
        dataset = CIFAR100(cfg)

    else:
        raise("invalid dataset")
    
    return dataset