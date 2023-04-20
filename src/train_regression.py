import torch
import torch.nn.functional as F
import torch.optim as optim
import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb

from model_merging.model import MLP_regression
from model_merging.curvature_regression import compute_and_store_fisher_diagonals, compute_and_store_gradients
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

    train_losses = []
    val_losses = []

    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0
        for _, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x.to(device))
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss


        train_loss = train_loss.detach()/len(train_loader)
        train_losses.append(train_loss)

        print(
            f"Epoch [{epoch + 1}/{cfg.train.epochs}], Training Loss: {train_loss/len(train_loader):.4f}"
        )

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for _, (x, y) in enumerate(test_loader):
                out = model(x.to(device))
                loss = criterion(out,y)
                val_loss += loss
            print(
                f"Epoch [{epoch + 1}/{cfg.train.epochs}], Validation Loss: {val_loss/len(test_loader):.4f}"
            )
            val_loss /= len(test_loader)
            val_losses.append(val_loss)

    if cfg.train.plot:
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    print("")
    torch.save(model.state_dict(), name)
    print(name)

    return name


def inference(model, test_loader, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_data = []
    x_data = []
    y_data = []

    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            out = model(x.to(device))
            loss = criterion(out, y)
            loss_data.append(loss)
            x_data.append(np.array(x.flatten()))
            y_data.append(np.array(out.flatten())) 

    return np.array(x_data).flatten(), np.array(y_data).flatten(), sum(loss_data)/len(test_loader)


@hydra.main(config_path="./configurations", config_name="train_regression.yaml")
def main(cfg):
    if cfg.train.torch_seed > -1:
        torch.manual_seed(cfg.train.torch_seed)

    dataset = create_dataset(cfg)
    train_loader, test_loader = dataset.create_dataloaders()
    model = MLP_regression(cfg)
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    criterion = torch.nn.MSELoss()

    name = "{}{}_epoch{}.pt".format(
        cfg.data.model_path,
        cfg.data.dataset,
        cfg.train.epochs,
    )

    name = train(cfg, name, train_loader, test_loader, model, optimizer, criterion)
    
    test_loader = dataset.create_inference_dataloader()
    inference(cfg, model, test_loader, criterion)

    if cfg.train.fisher_diagonal:
        compute_and_store_fisher_diagonals(cfg, name)

    if cfg.train.fisher_gradients:
        compute_and_store_gradients(cfg, name)


if __name__ == "__main__":
    main()
