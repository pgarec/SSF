import torch
import torch.nn.functional as F
import torch.optim as optim
import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb

from model_merging.model import MLP
from model_merging.fisher import compute_fisher_diags, compute_fisher_grads
from model_merging.data import create_dataset

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def train(cfg, name, train_loader, test_loader, model, optimizer, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    y_classes = dict(zip(cfg.data.classes, range(len(cfg.data.classes))))

    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0
        for _, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x.to(device))
            batch_onehot = y.apply_(lambda x: y_classes[x])
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            
            loss.backward()
            optimizer.step()
            train_loss += loss
            # wandb.log({"Training loss": loss/len(train_loader)})

        print(
            f"Epoch [{epoch + 1}/{cfg.train.epochs}], Training Loss: {train_loss/len(train_loader):.4f}"
        )

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for _, (x, y) in enumerate(test_loader):
                out = model(x.to(device))
                batch_onehot = y.apply_(lambda x: y_classes[x])
                loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
                val_loss += loss
            # wandb.log({"Validation loss": loss/len(val_loader)})
            print(
                f"Epoch [{epoch + 1}/{cfg.train.epochs}], Validation Loss: {val_loss/len(test_loader):.4f}"
            )

    print("")
    torch.save(model.state_dict(), name)
    print(name)

    return name


def inference(cfg, name, test_loader, criterion):
    model = MLP(cfg)
    model.load_state_dict(torch.load(name))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avg_loss = [0] * len(cfg.data.classes)
    count = [0] * len(cfg.data.classes)
    y_classes = dict(zip(cfg.data.classes, range(len(cfg.data.classes))))

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = model(x.to(device))
            batch_onehot = y.apply_(lambda i: y_classes[i])
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            avg_loss[y] += loss.item()
            count[y] += 1

            if batch_idx == 0 and cfg.train.plot_sample:
                probs = torch.softmax(out[0], dim=-1)
                plt.bar(np.arange(len(cfg.data.classes)), probs.cpu().numpy())
                plt.xlabel("Number of classes")
                plt.ylabel("Class probabilities for y={}".format(y))
                plt.xticks(np.arange(len(cfg.data.classes)))
                plt.show()
            
    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.classes))]

    if cfg.train.plot:
        plt.bar(list(y_classes.keys()), avg_loss)
        plt.xlabel("Number of classes")
        plt.ylabel("Average Test Loss")
        plt.xticks(list(y_classes.keys()))
        plt.show()

        print("")


def train_subsets(cfg):
    subset_length = cfg.data.subset_length
    classes = cfg.data.classes
    assert subset_length < len(classes)
    subsets = [classes[i:i+subset_length] for i in range(0, len(classes), subset_length)]

    for subset in subsets:
        cfg_subset = cfg
        cfg_subset.data.classes = subset
        cfg_subset.data.n_classes = len(subset)
        dataset = create_dataset(cfg_subset)
        train_loader, test_loader = dataset.create_dataloaders(unbalanced=cfg_subset.data.unbalanced)
        
        model = MLP(cfg_subset)
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.train.lr, momentum=cfg_subset.train.momentum
        )
        criterion = torch.nn.CrossEntropyLoss()

        if cfg.data.unbalanced == []:
            name = "{}{}_{}_epoch{}.pt".format(
                cfg.data.model_path,
                cfg.data.dataset,
                "".join(map(str, cfg.data.classes)), cfg.train.epoch
            )
        else:
            name = "{}{}_{}_epoch{}_{}.pt".format(
                cfg.data.model_path,
                cfg.data.dataset,
                "".join(map(str, cfg.data.classes)), cfg.train.epoch,
                "".join(map(str, cfg.data.unbalanced))
        )

        name = train(cfg_subset, name, train_loader, test_loader, model, optimizer, criterion)
        
        test_loader = dataset.create_inference_dataloader()
        inference(cfg_subset, name, test_loader, criterion)

        if cfg_subset.train.fisher:
            cfg_subset.train.name = name
            compute_fisher_diags(cfg_subset)
 

@hydra.main(config_path="./configurations", config_name="train.yaml")
def main(cfg):
    if cfg.train.torch_seed > -1:
        torch.manual_seed(cfg.train.torch_seed)

    if cfg.data.subset_length > 0:
        train_subsets(cfg)
    
    else:
        dataset = create_dataset(cfg)
        train_loader, test_loader = dataset.create_dataloaders(unbalanced=cfg.data.unbalanced)
        model = MLP(cfg)
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        if cfg.data.unbalanced == []:
            name = "{}{}_{}_epoch{}.pt".format(
                cfg.data.model_path,
                cfg.data.dataset,
                "".join(map(str, cfg.data.classes)), cfg.train.epochs
            )
        else:
            name = "{}{}_{}_epoch{}_{}.pt".format(
                cfg.data.model_path,
                cfg.data.dataset,
                "".join(map(str, cfg.data.classes)), cfg.train.epochs,
                "".join(map(str, cfg.data.unbalanced)))

        name = train(cfg, name, train_loader, test_loader, model, optimizer, criterion)
        
        test_loader = dataset.create_inference_dataloader()
        inference(cfg, name, test_loader, criterion)

        if cfg.train.fisher_diagonal:
            compute_fisher_diags(cfg, name)

        if cfg.train.fisher_gradients:
            compute_fisher_grads(cfg, name)


if __name__ == "__main__":
    main()