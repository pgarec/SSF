import torch
import hydra
from model_merging.data import MNIST, load_models
from merge_fisher import evaluate_fisher
from merge_isotropic import evaluate_isotropic
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


def plot_results(results):
    plt.bar(list(results.keys()), list(results.values()))
    plt.xlabel("Type of merging")
    plt.ylabel("Average Test Loss")
    plt.xticks(list(results.keys()))
    plt.show()


def plot_results_number(isotropic_loss, fisher_loss, output_loss):

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Loss for each digit across merging techniques", fontsize=16)
    
    # Plot the loss for each digit in each model
    width = 0.25
    labels = ["Isotropic", "Fisher", "Output"]

    for digit in range(len(isotropic_loss)):
        #ax.set_title(f"Digit {digit}")
        ax.bar(digit, isotropic_loss[digit], label=f"Isotropic", color="b", width=0.25)
        ax.bar(digit + width, fisher_loss[digit], label=f"Fisher", color="g", width=0.25)
        ax.bar(digit + 2*width, output_loss[digit], label=f"Output", color="r", width=0.25)
        
    ax.set_xticks(np.arange(len(isotropic_loss)) + width / 2)
    ax.set_xticklabels([str(digit) for digit in range(10)])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Loss")
    ax.legend(labels)
    plt.show()


@hydra.main(config_path="./configurations", config_name="merge.yaml")
def evaluate_techniques(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(cfg)
    dataset = MNIST(cfg)
    cfg.data.batch_size_test = 1
    test_loader = dataset.create_inference_dataloader()
    y_classes = dict(zip(cfg.data.digits, range(len(cfg.data.digits))))
    criterion = torch.nn.CrossEntropyLoss()
    outputs = []

    isotropic_loss, count = evaluate_isotropic(cfg, models, test_loader, criterion)
    fisher_loss, count = evaluate_fisher(cfg, models, test_loader, criterion)

    output_loss = [0] * len(cfg.data.digits)
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            batch_onehot = y.apply_(lambda i: y_classes[i])
            output_model = []
            for m in models:
                m.to(device)
                m.eval()
                out = m(x.to(device))
                output_model.append(out)
            
            outputs = torch.stack(output_model, dim=0).sum(dim=0) / len(models)
            loss = criterion(outputs, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            output_loss[y] += loss

    loss = sum(output_loss) / len(test_loader)

    results = {'isotropic loss': sum(isotropic_loss)/len(test_loader), 'fisher_loss': sum(fisher_loss)/len(test_loader), 'output_loss': loss}
    plot_results(results)
    print(results)
    
    isotropic_loss_avg = [isotropic_loss[i] / count[i] for i in range(len(cfg.data.digits))]
    fisher_loss_avg = [fisher_loss[i] / count[i] for i in range(len(cfg.data.digits))]
    output_loss_avg = [output_loss[i] / count[i] for i in range(len(cfg.data.digits))]
    plot_results_number(isotropic_loss_avg, fisher_loss_avg, output_loss_avg)



if __name__ == "__main__":
    evaluate_techniques()
