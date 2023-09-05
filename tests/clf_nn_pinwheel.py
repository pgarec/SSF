import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import omegaconf
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.model_merging.curvature import compute_fisher_diagonals, compute_gradients
from src.model_merging.datasets.pinwheel import make_pinwheel_data
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.merge_permutation import merging_models_permutation
from src.model_merging.permutation import scaling_permutation
from src.model_merging.permutation import random_weight_permutation

# Load configuration
cfg = omegaconf.OmegaConf.load('./configurations/perm_pinwheel.yaml')
seed = cfg.train.torch_seed

if seed > -1:
    np.random.seed(seed)
    torch.manual_seed(seed)

# Constants
num_clusters = 5
samples_per_cluster = 2000
K = 15
N = 2
P = 2
H = 16
plot = False

# Define Model class
class Model(nn.Module):
    def __init__(self, num_features, H, num_output, torch_seed=-1):
        super(Model, self).__init__()

        if torch_seed > -1:
            torch.manual_seed(torch_seed)

        self.model = nn.Sequential(
            nn.Linear(num_features, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, num_output, bias=False)
        )

    def forward(self, x):
        x = self.model(x)

        return x

    def get_trainable_parameters(self):

        return [param for param in self.parameters() if param.requires_grad]
    

# Clone Model function
def clone_model(model, num_features, H, num_output, seed):
    cloned = model.__class__(num_features, H, num_output, seed)
    cloned.load_state_dict(model.state_dict().copy())

    return cloned


# Evaluate Model function
def evaluate_model(model, val_loader, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_loss = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            out = model(x.to(device))
            loss = criterion(out, y)
            avg_loss += loss

    return avg_loss / len(val_loader)


# Create DataLoader
def create_dataloader(data, labels, batch_size):
    dataset = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels).long().reshape(-1))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Main Experiment
def main_experiment():
    sns.set_style('darkgrid')
    palette = sns.color_palette('colorblind')

    batch_data = True
    data, labels = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

    X_train = data[:350]
    X_valid = data[350:400]
    X_test = data[400:]

    y_train = labels[:350].astype('int32')
    y_valid = labels[350:400].astype('int32')
    y_test = labels[400:].astype('int32')

    X_train = torch.from_numpy(X_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    X_test = torch.from_numpy(X_test).float()

    y_train = torch.from_numpy(y_train).long().reshape(-1)
    y_valid = torch.from_numpy(y_valid).long().reshape(-1)
    y_test = torch.from_numpy(y_test).long().reshape(-1)

    if batch_data:
        train_loader = create_dataloader(X_train, y_train, batch_size=100)
        val_loader = create_dataloader(X_valid, y_valid, batch_size=100)

    num_features = X_train.shape[-1]
    num_output = num_clusters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    n_models = 3
    max_epoch = 100

    for m in range(n_models):
        model = Model(num_features, H, num_output, seed)
        lr = cfg.train.lr * ((m + 1) * 0.5)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=cfg.train.weight_decay)
        criterion = nn.CrossEntropyLoss(reduction='sum')

        best_valid_accuracy = 0
        max = max_epoch * (m + 1)

        for epoch in range(max):
            train_loss = 0
            for _, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                out = model(x.to(device))
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss
            print(f"Epoch [{epoch + 1}/{max}], Training Loss: {train_loss / len(train_loader):.4f}")

            val_loss = 0
            with torch.no_grad():
                model.eval()
                for _, (x, y) in enumerate(val_loader):
                    out = model(x.to(device))
                    loss = criterion(out, y)
                    val_loss += loss

                print(f"Epoch [{epoch + 1}/{max}], Validation Loss: {val_loss / len(val_loader):.4f}")

        models.append(model)

    cfg.data.n_examples = 350

    parameters = models[0].get_trainable_parameters()
    metatheta = nn.utils.parameters_to_vector(parameters)
    print("Number of parameters: {}".format(len(metatheta)))

    for n, model in enumerate(models):
        print("Loss model {}:{}".format(n, evaluate_model(model, val_loader, criterion)))

    output_model = clone_model(models[0], num_features, H, num_output, seed)
    isotropic_model = merging_models_isotropic(output_model, models)
    print("Isotropic model loss: {}".format(evaluate_model(isotropic_model, val_loader, criterion)))

    output_model = clone_model(models[0], num_features, H, num_output, seed)
    fishers = [compute_fisher_diagonals(m, train_loader, num_clusters, cfg.data.n_examples) for m in models]
    fisher_model = merging_models_fisher(output_model, models, fishers)
    print("Fisher model loss: {}".format(evaluate_model(fisher_model, val_loader, criterion)))

    metamodel = isotropic_model
    grads = [compute_gradients(m, train_loader, num_clusters, cfg.data.n_examples) for m in models]
    cfg.train.initialization = "MLP"
    cfg.data.n_classes = num_clusters
    output_model = clone_model(models[0], num_features, H, num_output, seed)
    metamodel = Model(num_features, H, num_output, seed)
    perm_model, _, _ = merging_models_permutation(cfg, metamodel, models, grads, fishers, val_loader, criterion, plot=True)
    print("Permutation model loss: {}".format(evaluate_model(perm_model, val_loader, criterion)))


if __name__ == "__main__":
    main_experiment()