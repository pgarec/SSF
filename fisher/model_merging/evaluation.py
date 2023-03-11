import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def evaluate_metamodel(cfg, merged_model, criterion, test_loader):
    avg_loss = [0] * len(cfg.data.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = [0] * len(cfg.data.classes)
    merged_model.to(device)
    y_classes = dict(zip(cfg.data.classes, range(len(cfg.data.classes))))

    with torch.no_grad():
        merged_model.eval()
        for _, (x, y) in enumerate(test_loader):
            out = merged_model(x.to(device))
            batch_onehot = y.apply_(lambda i: y_classes[i])
            count[y] += 1
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            avg_loss[y] += loss
    
    return avg_loss, count


def evaluate_minimodels(cfg, models, criterion, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_loss_models = [[0] * len(cfg.data.classes) for m in models]
    count = [0] * len(cfg.data.classes)
    y_classes = dict(zip(cfg.data.classes, range(len(cfg.data.classes))))

    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            # batch size 1 for evaluation
            batch_onehot = y.apply_(lambda i: y_classes[i])
            count[y] += 1

            for m in range(len(models)):
                out = models[m](x.to(device))
                loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
                avg_loss_models[m][y] += loss.item()

        return avg_loss_models, count


def plot(cfg, avg_loss, avg_loss_models, count, models):
    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.classes))]
    print("Avg_loss merged model: {}".format(avg_loss))
    
    plt.bar(np.arange(len(cfg.data.classes)), avg_loss)
    plt.xlabel("Number of classes")
    plt.ylabel("Average Test Loss")
    plt.xticks(np.arange(len(cfg.data.classes)))
    plt.show()

    for m in range(len(models)):
        avg_loss = [avg_loss_models[m][i] / count[i] for i in range(len(cfg.data.classes))]
        print("Avg_loss model {}: {}".format(m, avg_loss))

        plt.bar(np.arange(len(cfg.data.classes)), avg_loss)
        plt.xlabel("Number of classes")
        plt.ylabel("Average Test Loss model {}".format(m))
        plt.xticks(np.arange(len(cfg.data.classes)))
        plt.show()


def plot_avg_merging_techniques(results):
    plt.bar(list(results.keys()), list(results.values()))
    plt.xlabel("Type of merging")
    plt.ylabel("Average Test Loss")
    plt.xticks(list(results.keys()))
    plt.show()


def plot_merging_techniques(cfg, isotropic_loss, fisher_loss, perm_loss, output_loss):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Loss for each digit across merging techniques", fontsize=16)
    
    width = 0.25
    labels = ["Isotropic", "Fisher", "Output"]

    for digit in range(len(isotropic_loss)):
        ax.bar(digit, isotropic_loss[digit], label=f"Isotropic", color="b", width=0.25)
        ax.bar(digit + width, fisher_loss[digit], label=f"Fisher", color="g", width=0.25)
        ax.bar(digit + 2*width, perm_loss[digit], label=f"Permutation", color="r", width=0.25)
        ax.bar(digit + 3*width, output_loss[digit], label=f"Output", color="y", width=0.25)
        
    ax.set_xticks(np.arange(len(isotropic_loss)) + width / 2)
    ax.set_xticklabels([str(digit) for digit in range(len(cfg.data.classes))])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Loss")
    ax.legend(labels)
    plt.show()
