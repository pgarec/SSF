import torch
import hydra
from model_merging.data import MNIST
from model_merging.model import MLP
from model_merging.data import load_fisher
from model_merging.merging import create_pairwise_grid_coeffs, create_random_coeffs 
from model_merging.merging import merging_models
import torch.nn.functional as F


def load_models(cfg):
    models = []

    for model_name in cfg.models:
        model = MLP(cfg)
        model.load_state_dict(torch.load(cfg.data.model_path+cfg.models[model_name]+".pt"))
        models.append(model)

    return models


def load_fishers(cfg):
    fishers = []

    for model_name in cfg.models:
        split = cfg.models[model_name].split("_")
        model_name = "_".join(split[:2]+[split[-1]])
        path = cfg.data.fisher_path + model_name
        fisher = load_fisher(path)
        fishers.append(fisher)

    return fishers


def get_coeffs_set(cfg):
    n_models = len(cfg.merge.n_models)
    if cfg.merge.coeff_mode == "grid":
        assert n_models == 2
        return create_pairwise_grid_coeffs(cfg.merge.n_coeffs)
    elif cfg.merge.coeff_mode == "random":
        return create_random_coeffs(n_models, cfg.merge.n_coeffs)
    else:
        raise ValueError


def evaluate(cfg, merged_model, models, criterion, test_loader):
    val_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged_model.to(device)
    min_class = min(cfg.data.digits)

    with torch.no_grad():
        merged_model.eval()
        for _, (x, y) in enumerate(test_loader):
            out = merged_model(x.to(device))
            loss = criterion(out, F.one_hot(y-min_class, cfg.data.n_classes).to(torch.float))
            val_loss += loss

        print(f"Validation Loss: {val_loss/len(test_loader):.4f}")


@hydra.main(config_path="./configurations", config_name="merge.yaml")
def main(cfg):
    models = load_models(cfg)
    fishers = load_fishers(cfg)
    dataset = MNIST(cfg)
    _, test_loader = dataset.create_dataloaders()
    criterion = torch.nn.CrossEntropyLoss()

    merged_model = merging_models(cfg, models, fishers)
    evaluate(cfg, merged_model, models, criterion, test_loader)

if __name__ == "__main__":
    main()
