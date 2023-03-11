import torch
import pickle
from .model import MLP
from .datasets.mnist import MNIST
from .datasets.fmnist import FashionMNIST
from .datasets.cifar import CIFAR10, CIFAR100


def load_models(cfg, names = []):
    models = []

    if names == []:
        for model_name in cfg.models:
            model = MLP(cfg)
            model.load_state_dict(torch.load(cfg.data.model_path+cfg.models[model_name]+".pt"))
            models.append(model)
    
    else:
        for model_name in names:
            model = MLP(cfg)
            model.load_state_dict(torch.load(cfg.data.model_path+model_name+".pt"))
            models.append(model)

    return models


def store_file(file, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(file, f)


def load_file(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def load_fishers(cfg, names=[]):
    fishers = []

    if names == []:
        for model_name in cfg.models:
            path = cfg.data.fisher_path + cfg.models[model_name]
            fisher = load_file(path)
            fishers.append(fisher)

    else:
        for model_name in names:
            path = cfg.data.fisher_path + model_name
            fisher = load_file(path)
            fishers.append(fisher)
    
    return fishers


def load_grads(cfg, names=[]):
    grads = []

    if names == []:
        for model_name in cfg.models:
            path = cfg.data.grad_path + cfg.models[model_name]
            grad = load_file(path)
            grads.append(grad)
    
    else:
        for model_name in names:
            path = cfg.data.grad_path + model_name
            grad = load_file(path)
            grads.append(grad)

    return grads


def create_dataset(cfg):
    if cfg.data.dataset == "MNIST":
        assert cfg.data.image_shape == 784

        dataset = MNIST(cfg)

    elif cfg.data.dataset == "FashionMNIST":  
        assert cfg.data.image_shape == 784

        dataset = FashionMNIST(cfg)
    
    elif cfg.data.dataset == "CIFAR10":
        assert cfg.data.image_shape == 3072

        dataset = CIFAR10(cfg)
    
    elif cfg.data.dataset == "CIFAR100":
        assert cfg.data.image_shape == 3072
        cfg.data.classes = list(range(0,100))

        dataset = CIFAR100(cfg)

    else:
        raise("invalid dataset")
    
    return dataset