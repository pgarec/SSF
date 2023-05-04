'''
File with the experiment on the pinwheel dataset.
Generating plot with confidence, where we plot conf = np.max(preds, axis=1).
Compare this with Laplace approximation.
In addition to that, we should measure some classification metrics, like ECE, Brier score, and accuracy.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("stochman")

import matplotlib.colors as colors
import seaborn as sns
from torch import nn
# from manifold import cross_entropy_manifold
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from src.model_merging.datasets.pinwheel import make_pinwheel_data
import hydra
from src.model_merging.curvature import compute_fisher_diagonals, compute_gradients
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.merge_permutation import merging_models_permutation, merging_models_weight_permutation, merging_models_scaling_permutation
from src.model_merging.permutation import compute_permutations, l2_permutation, implement_permutation, implement_permutation_grad
from src.model_merging.permutation import scaling_permutation, random_weight_permutation
import random
import omegaconf

# CONFIGURATION
cfg = omegaconf.OmegaConf.load('./configurations/perm_pinwheel.yaml')
seed = cfg.train.torch_seed

if seed > -1:
    np.random.seed(seed)
    torch.manual_seed(seed)

num_clusters = 5        # number of clusters in pinwheel data
samples_per_cluster = 1000  # number of samples per cluster in pinwheel
K = 15                     # number of components in mixture model
N = 2                      # number of latent dimensions
P = 2                      # number of observation dimensions
H = 16
plot = False

class Model(nn.Module):
    def __init__(self, num_features, H, num_output, torch_seed=-1):
        super(Model, self).__init__()

        if torch_seed > -1:
            torch.manual_seed(torch_seed)

        self.model = nn.Sequential(
            nn.Linear(num_features, H),
            # torch.nn.ReLU(),
            torch.nn.Tanh(),
            nn.Linear(H, H),
            # torch.nn.ReLU(),
            torch.nn.Tanh(),
            nn.Linear(H,num_output, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        
        return x
    
    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]
    
    def get_trainable_linear_parameters(self):
        for p in self.named_parameters():
            print(p)
        return [param for param in self.named_parameters() if param.requires_grad and isinstance(param, nn.Linear)]


def clone_model(model, num_features, H, num_output, seed):
    cloned = model.__class__(num_features, H, num_output, seed)
    cloned.load_state_dict(model.state_dict().copy())
        
    return cloned   


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


def evaluate_model_in_depth(model, val_loader, criterion, plot=False):
    avg_loss = [0] * num_clusters
    count = [0] * num_clusters
    y_classes = dict(zip(range(num_clusters), range(num_clusters)))

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            out = model(x.to(device))
            loss = criterion(out, y)
            avg_loss[y] += loss
            count[y] += 1

    if plot:
        avg_loss_mean = [x/y for (x,y) in zip(avg_loss,count)]
        plt.bar(list(y_classes.keys()), avg_loss_mean)
        plt.xlabel("Number of classes")
        plt.ylabel("Average Test Loss")
        plt.xticks(list(y_classes.keys()))
        plt.show()


sns.set_style('darkgrid')
palette = sns.color_palette('colorblind')


batch_data = True
data, labels = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

# define train and validation 
X_train = data[:350]
X_valid = data[350:400]
X_test = data[400:]

y_train = labels[:350].astype('int32')
y_valid = labels[350:400].astype('int32')
y_test = labels[400:].astype('int32')

one_hot_yvtest = np.zeros((y_test.size, y_test.max() + 1))
one_hot_yvtest[np.arange(y_test.size), y_test.reshape(-1)] = 1

X_train = torch.from_numpy(X_train).float()
X_valid = torch.from_numpy(X_valid).float()
X_test = torch.from_numpy(X_test).float()

y_train = torch.from_numpy(y_train).long().reshape(-1)
y_valid = torch.from_numpy(y_valid).long().reshape(-1)
y_test = torch.from_numpy(y_test).long().reshape(-1)

if batch_data:
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True)

if plot:
    plt.scatter(X_train[:,0], X_train[:,1], s=40, c=y_train, cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
    plt.title("Train data")
    plt.show()

    plt.scatter(X_valid[:,0], X_valid[:,1], s=40, c=y_valid, cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
    plt.title("Validation data")
    plt.show()

num_features = X_train.shape[-1]
print(num_features)
num_output = num_clusters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = []
n_models = 3
max_epoch = 100

for m in range(n_models):
    model = Model(num_features, H, num_output, seed)
    lr = cfg.train.lr*((m+1)*0.5)

    optimizer = torch.optim.SGD(model.parameters(),  lr=lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    best_valid_accuracy = 0
    max = max_epoch*(m+1)

    for epoch in range(max):
        train_loss = 0
        for _, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x.to(device))
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss
        print(
            f"Epoch [{epoch + 1}/{max}], Training Loss: {train_loss/len(train_loader):.4f}"
        )

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for _, (x, y) in enumerate(val_loader):
                out = model(x.to(device))
                loss = criterion(out, y)
                val_loss += loss

            print(
                f"Epoch [{epoch + 1}/{max}], Validation Loss: {val_loss/len(val_loader):.4f}"
            )
    
    models.append(model)

print("m {}".format(cfg.data.m))

parameters = models[0].get_trainable_parameters()
metatheta = nn.utils.parameters_to_vector(parameters)
print("Number of parameters: {}".format(len(metatheta)))

for n, model in enumerate(models):
    print("Loss model {}:{}".format(n, evaluate_model(model, val_loader, criterion)))

output_model = clone_model(models[0], num_features, H, num_output, seed)
isotropic_model = merging_models_isotropic(output_model, models)
print("Istropic model loss: {}".format(evaluate_model(isotropic_model, val_loader, criterion)))

output_model = clone_model(models[0], num_features, H, num_output, seed)
fishers = [compute_fisher_diagonals(m, train_loader, num_clusters) for m in models]
fisher_model = merging_models_fisher(output_model, models, fishers)
print("Fisher model loss: {}".format(evaluate_model(fisher_model, val_loader, criterion)))

metamodel = isotropic_model
grads = [compute_gradients(m, train_loader, num_clusters) for m in models]
cfg.data.n_examples = 350
cfg.train.initialization = "MLP"
cfg.data.n_classes = num_clusters
output_model = clone_model(models[0], num_features, H, num_output, seed)
# metamodel = merging_models_isotropic(output_model, models)
metamodel = Model(num_features, H, num_output, seed)
perm_model = merging_models_permutation(cfg, metamodel, models, grads, val_loader, criterion, plot=True)
print("Permutation model loss: {}".format(evaluate_model(perm_model, val_loader, criterion)))

# metamodel = models[0]
# metamodel = isotropic_model
# cfg.data.n_examples = 350
# metamodel = Model(num_features, H, num_output, seed)
# grads = [compute_gradients(m, train_loader, num_clusters) for m in models]
# permutations = compute_permutations(models, cfg.data.layer_weight_permutation, cfg.data.weight_permutations)
# weight_symmetries_model = merging_models_weight_permutation(cfg, metamodel, models, permutations, grads, val_loader, criterion, plot=True)
# print("Weight permutation model loss: {}".format(evaluate_model(weight_symmetries_model, val_loader, criterion)))

# metamodel = models[0]
# metamodel = isotropic_model 
# # metamodel = Model(num_features, H, num_output, seed)
# grads = [compute_gradients(m, train_loader, num_clusters) for m in models]
# perm_model = merging_models_scaling_permutation(cfg, metamodel, models, grads, val_loader, criterion, plot=True)
# print("Scaling permutation model loss: {}".format(evaluate_model(perm_model, val_loader, criterion)))