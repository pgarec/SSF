# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CROMAI  ---  (pol.garcia@bsc.es)
# Barcelona Supercomputing Center (BSC)

import torch
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as colors
import seaborn as sns
from torch import nn
# from manifold import cross_entropy_manifold
from src.model_merging.datasets.pinwheel import make_pinwheel_data
from src.model_merging.curvature import fim_diag, grad_diag
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.merge_permutation import merging_models_permutation
import omegaconf

font = {'family' : 'serif',
        'size'   : 20,
        'weight': 'bold'}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

color_palette_1 = ['#335c67','#fff3b0','#e09f3e','#9e2a2b','#540b0e']
color_palette_2 = ['#177e89','#084c61','#db3a34','#ef8354','#323031']
color_palette_3 = ['#bce784','#5dd39e','#348aa7','#525274','#513b56']
color_palette_4 = ['#002642','#840032','#e59500','#e5dada','#02040e']
color_palette_5 = ['#202c39','#283845','#b8b08d','#f2d449','#f29559']

palette_red = ["#03071e","#370617","#6a040f","#9d0208","#d00000","#dc2f02","#e85d04","#f48c06","#faa307","#ffba08"]
palette_blue = ["#012a4a","#013a63","#01497c","#014f86","#2a6f97","#2c7da0","#468faf","#61a5c2","#89c2d9","#a9d6e5"]
palette_green = ['#99e2b4','#88d4ab','#78c6a3','#67b99a','#56ab91','#469d89','#358f80','#248277','#14746f','#036666']
palette_pink = ["#ea698b","#d55d92","#c05299","#ac46a1","#973aa8","#822faf","#6d23b6","#6411ad","#571089","#47126b"]
palette_super_red = ["#641220","#6e1423","#85182a","#a11d33","#a71e34","#b21e35","#bd1f36","#c71f37","#da1e37","#e01e37"]

palette = color_palette_5
meta_color = 'r'

# CONFIGURATION
cfg = omegaconf.OmegaConf.load('./configurations/perm_pinwheel.yaml')
seed = cfg.train.torch_seed

if seed > -1:
    np.random.seed(seed)
    torch.manual_seed(seed)

num_clusters = 5        # number of clusters in pinwheel data
samples_per_cluster = 2000  # number of samples per cluster in pinwheel
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
            torch.nn.ReLU(),
            nn.Linear(H, H),
            torch.nn.ReLU(),
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

    return avg_loss / (len(val_loader))


if __name__ == "__main__":
    plot = True
    sns.set_style('darkgrid')
    palette = sns.color_palette('colorblind')

    batch_data = True
    data, labels = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25, seed)

    print("Torch seed {}".format(cfg.train.torch_seed))
    print("X data shape {}".format(data.shape))

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
        plt.figure(figsize=(8, 8))  # Set the figure size to a square aspect ratio
        ax = plt.gca()  # Get the current Axes
        ax.set_facecolor('white')
        plt.scatter(X_train[:, 0], X_train[:, 1], s=70, c=y_train, cmap=colors.ListedColormap(palette[:5]))
        plt.xticks([])
        plt.yticks([])
        plt.title("Pinwheel Data", family='serif', fontsize=20)
        plt.rc('text', usetex=True)
        plt.rc('font', **font)
        plt.rc('text.latex', preamble=r'\usepackage{bm}')


        plt.show()
        # plt.scatter(X_train[:,0], X_train[:,1], s=40, c=y_train, cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
        # plt.title("Pinwheel data")
        # plt.show()

        # plt.scatter(X_valid[:,0], X_valid[:,1], s=40, c=y_valid, cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
        # plt.title("Validation data")
        # plt.show()

    num_features = X_train.shape[-1]
    print("Num features {}".format(num_features))
    num_output = num_clusters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    n_models = 3
    max_epoch = 100

    for m in range(1,n_models):
        model = Model(num_features, H, num_output, seed)
        lr = cfg.train.lr*((m+1)*0.5)

        optimizer = torch.optim.SGD(model.parameters(),  lr=lr, weight_decay=cfg.train.weight_decay)
        criterion = nn.CrossEntropyLoss()

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


            val_loss = 0
            with torch.no_grad():
                model.eval()
                for _, (x, y) in enumerate(val_loader):
                    out = model(x.to(device))
                    loss = criterion(out, y)
                    val_loss += loss

                # print(
                #     f"Epoch [{epoch + 1}/{max}], Validation Loss: {val_loss/len(val_loader):.4f}"
                # )
        
        models.append(model)

    print("m {}".format(cfg.data.m))
    print(val_loader)
    cfg.data.n_examples = 350

    parameters = models[0].get_trainable_parameters()
    metatheta = nn.utils.parameters_to_vector(parameters)
    print("Number of parameters: {}".format(len(metatheta)))

    for n, model in enumerate(models):
        print("Loss model {}:{}".format(n, evaluate_model(model, val_loader, criterion)))

    output_model = clone_model(models[0], num_features, H, num_output, seed)
    isotropic_model = merging_models_isotropic(output_model, models)
    print("Istropic model loss: {}".format(evaluate_model(isotropic_model, val_loader, criterion)))

    output_model = clone_model(models[0], num_features, H, num_output, seed)
    fishers = [fim_diag(m, val_loader, cfg.data.n_examples) for m in models]
    fisher_model = merging_models_fisher(output_model, models, fishers)
    print("Fisher model loss: {}".format(evaluate_model(fisher_model, val_loader, criterion)))

    grads = [grad_diag(m, train_loader, cfg.data.n_examples) for m in models]
    cfg.train.initialization = "MLP"
    cfg.data.n_classes = num_clusters
    output_model = clone_model(models[0], num_features, H, num_output, seed)
    metamodel = Model(num_features, H, num_output, seed)
    perm_model, _, _ = merging_models_permutation(cfg, metamodel, models, grads, fishers, val_loader, llm=False, criterion=criterion, plot=True, store=True)
    print("Permutation model loss: {}".format(evaluate_model(perm_model, val_loader, criterion)))

