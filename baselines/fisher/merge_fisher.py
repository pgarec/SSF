import torch
import hydra
from model_merging.data import MNIST
from model_merging.model import MLP
from model_merging.data import load_fisher
from model_merging.merging import create_pairwise_grid_coeffs, create_random_coeffs 
from model_merging.merging import merging_models_fisher, merging_models_isotropic
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


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

    avg_loss = [0] * len(cfg.data.digits)
    count = [0] * len(cfg.data.digits)

    avg_loss_models = [[0] * len(cfg.data.digits) for m in models]
    count_models = [[0] * len(cfg.data.digits) for m in models]

    with torch.no_grad():
        merged_model.eval()
        for _, (x, y) in enumerate(test_loader):
            out = merged_model(x.to(device))
            loss = criterion(out, F.one_hot(y-min_class, cfg.data.n_classes).to(torch.float))
            avg_loss[y[0].item()] += loss.item()
            count[y[0].item()] += 1
            val_loss += loss

            for m in range(len(models)):
                out = models[m](x.to(device))
                loss = criterion(out, F.one_hot(y-min_class, cfg.data.n_classes).to(torch.float))
                avg_loss_models[m][y[0].item()] += loss.item()
                count_models[m][y[0].item()] += 1

        print(f"Validation Loss: {val_loss/len(test_loader):.4f}")

    print("")
    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.digits))]
    print("Avg_loss merged model: {}".format(avg_loss))
    
    plt.bar(np.arange(len(cfg.data.digits)), avg_loss)
    plt.xlabel("Number of classes")
    plt.ylabel("Average Test Loss")
    plt.xticks(np.arange(len(cfg.data.digits)))
    plt.show()

    for m in range(len(models)):
        avg_loss = [avg_loss_models[m][i] / count[i] for i in range(len(cfg.data.digits))]
        print("Avg_loss model {}: {}".format(m, avg_loss))

        plt.bar(np.arange(len(cfg.data.digits)), avg_loss)
        plt.xlabel("Number of classes")
        plt.ylabel("Average Test Loss model {}".format(m))
        plt.xticks(np.arange(len(cfg.data.digits)))
        plt.show()

    print("")


@hydra.main(config_path="./configurations", config_name="merge.yaml")
def main(cfg):
    models = load_models(cfg)
    fishers = load_fishers(cfg)
    dataset = MNIST(cfg)
    _, test_loader = dataset.create_dataloaders()
    criterion = torch.nn.CrossEntropyLoss()

    merged_model = merging_models_fisher(cfg, models, fishers)
    evaluate(cfg, merged_model, models, criterion, test_loader)


if __name__ == "__main__":
    main()
