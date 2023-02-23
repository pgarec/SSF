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
    avg_loss = [0] * len(cfg.data.digits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = [0] * len(cfg.data.digits)
    merged_model.to(device)
    y_classes = dict(zip(cfg.data.digits, range(len(cfg.data.digits))))

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
    avg_loss_models = [[0] * len(cfg.data.digits) for m in models]
    count = [0] * len(cfg.data.digits)
    y_classes = dict(zip(cfg.data.digits, range(len(cfg.data.digits))))

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