import torch
import pickle
from .model import MLP, MLP_regression
from .datasets.mnist import MNIST
from .datasets.fmnist import FashionMNIST
from .datasets.cifar import CIFAR10, CIFAR100
from .datasets.snelson import SNELSON


def load_models(cfg, names=[]):
    models = []

    if names == []:
        for model_name in cfg.models:
            model = MLP(cfg)
            model.load_state_dict(torch.load(cfg.data.model_path+cfg.models[model_name]+".pt"))
            models.append(model)
    
    else:
        for model_name in names:
            model = MLP(cfg)
            name = model_name.split('/')[-1][:-3] 
            path = cfg.data.model_path + name + ".pt"
            model.load_state_dict(torch.load(path))
            models.append(model)

    return models


def load_models_regression(cfg, names=[]):
    models = []

    if names == []:
        for model_name in cfg.models:
            model = MLP_regression(cfg)
            model.load_state_dict(torch.load(cfg.data.model_path+cfg.models[model_name]+".pt"))
            models.append(model)
    
    else:
        for model_name in names:
            model = MLP_regression(cfg)
            name = model_name.split('/')[-1][:-3] 
            path = cfg.data.model_path + name + ".pt"
            model.load_state_dict(torch.load(path))
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
            name = model_name.split('/')[-1][:-3] 
            path = cfg.data.fisher_path + name
            fisher = load_file(path)            
            fishers.append(fisher)
    
    return fishers


def load_permutations(cfg, names=[]):
    perms = []

    if names == []:
        for model_name in cfg.models:
            path = cfg.data.perm_path + cfg.models[model_name]
            perm = load_file(path)
            perms.append(perm)

    else:
        for model_name in names:
            name = model_name.split('/')[-1][:-3] 
            path = cfg.data.perm_path + name
            perm = load_file(path)            
            perms.append(perm)
    
    return perms


def load_grads(cfg, names=[]):
    grads = []
    if names == []:
        for model_name in cfg.models:
            path = cfg.data.grad_path + cfg.models[model_name]
            grad = load_file(path)
            grads.append(grad)
    
    else:
        for model_name in names:
            name = model_name.split('/')[-1][:-3] 
            path = cfg.data.grad_path + name
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

    elif cfg.data.dataset == "SNELSON":
        dataset = SNELSON(cfg)

    else:
        raise("invalid dataset")
    
    return dataset