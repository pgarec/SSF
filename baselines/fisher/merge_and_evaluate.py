import torch
import hydra
from model_merging.data import MNIST
from model_merging.model import MLP
from model_merging.data import load_fisher


def load_models(cfg):
    models = []

    for model_name in cfg.models:
        model = MLP(cfg)
        model.load_state_dict(torch.load(model_name))
        models.append(model)

    return models


def load_fishers(cfg):
    fishers = []

    for model_name in cfg.models:
        model_name = model_name.split("_")[:1]
        path = cfg.data.fisher_path + model_name
        fisher = load_fisher(path)
        fishers.append(fisher)

    return fishers


@hydra.main(config_path="./configurations", config_name="merge.yaml")
def main(cfg):
    models = load_models(cfg)
    fishers = load_fishers(cfg)
    dataset = MNIST(cfg)
    train_loader, test_loader = dataset.create_dataloaders()
    criterion = torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    main()
