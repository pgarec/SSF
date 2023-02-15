import torch
import hydra
from model_merging.data import MNIST, load_models
from model_merging.model import MLP
from model_merging.merging import merging_models_isotropic
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def evaluate_metamodel(cfg, merged_model, criterion, test_loader):
    val_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged_model.to(device)
    y_classes = dict(zip(cfg.data.digits, range(len(cfg.data.digits))))

    with torch.no_grad():
        merged_model.eval()
        for _, (x, y) in enumerate(test_loader):
            out = merged_model(x.to(device))
            batch_onehot = y.apply_(lambda i: y_classes[i])
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            val_loss += loss

    avg_loss = val_loss / len(test_loader)
    print(avg_loss)
    
    return avg_loss


def evaluate(cfg, merged_model, models, criterion, test_loader):
    val_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged_model.to(device)

    avg_loss = [0] * len(cfg.data.digits)
    avg_loss_models = [[0] * len(cfg.data.digits) for m in models]
    count = [0] * len(cfg.data.digits)
    y_classes = dict(zip(cfg.data.digits, range(len(cfg.data.digits))))

    with torch.no_grad():
        merged_model.eval()
        for _, (x, y) in enumerate(test_loader):
            out = merged_model(x.to(device))
            batch_onehot = y.apply_(lambda i: y_classes[i])
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            avg_loss[y] += loss
            count[y] += 1
            val_loss += loss

            for m in range(len(models)):
                out = models[m](x.to(device))
                loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
                avg_loss_models[m][y] += loss

        print(f"Validation Loss: {val_loss/len(test_loader):.4f}")

    print("")
    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.digits))]
    print("Avg_loss merged model: {}".format(avg_loss))
    
    return avg_loss, avg_loss_models, count


def plot(cfg, avg_loss, avg_loss_models, count, models):
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


def evaluate_isotropic(cfg, models, test_loader, criterion):

    for m in models:
        avg_loss = evaluate_metamodel(cfg, m, criterion, test_loader)

    merged_model = merging_models_isotropic(cfg, models)
    avg_loss = evaluate_metamodel(cfg, merged_model, criterion, test_loader)

    return avg_loss


@hydra.main(config_path="./configurations", config_name="merge.yaml")
def isotropic(cfg):
    models = load_models(cfg)
    dataset = MNIST(cfg)
    _, test_loader = dataset.create_dataloaders()
    criterion = torch.nn.CrossEntropyLoss()

    merged_model = merging_models_isotropic(cfg, models)
    avg_loss, avg_loss_models, count = evaluate(cfg, merged_model, models, criterion, test_loader)

    if cfg.data.plot:
        plot(cfg, avg_loss, avg_loss_models, count, models)

    return avg_loss


if __name__ == "__main__":
    isotropic()
