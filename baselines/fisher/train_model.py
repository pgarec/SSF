import torch
import torch.nn.functional as F
import torch.optim as optim
import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb

from model_merging.data import MNIST
from model_merging.model import MLP
from compute_fisher import main as compute_fisher


def train(cfg, train_loader, test_loader, model, optimizer, criterion, unbalanced=False, unb_digits=[]):
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma
    # )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    y_classes = dict(zip(cfg.data.digits, range(len(cfg.data.digits))))

    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x.to(device))
            batch_onehot = y.apply_(lambda x: y_classes[x])
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            loss.backward()
            optimizer.step()
            train_loss += loss
            # scheduler.step()
            # wandb.log({"Training loss": loss/len(train_loader)})
            

        print(
            f"Epoch [{epoch + 1}/{cfg.train.epochs}], Training Loss: {train_loss/len(train_loader):.4f}"
        )

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for batch_idx, (x, y) in enumerate(test_loader):
                out = model(x.to(device))
                batch_onehot = y.apply_(lambda x: y_classes[x])
                loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
                val_loss += loss
            # wandb.log({"Validation loss": loss/len(val_loader)})
            print(
                f"Epoch [{epoch + 1}/{cfg.train.epochs}], Validation Loss: {val_loss/len(test_loader):.4f}"
            )

    print("")
    if cfg.data.unbalanced == []:
        name = "./models/mnist_{}_epoch{}.pt".format(
            "".join(map(str, cfg.data.digits)), epoch+1
        )
    else:
        name = "./models/mnist_{}_epoch{}_{}.pt".format(
            "".join(map(str, cfg.data.digits)), epoch+1,
            "".join(map(str, unb_digits))
        )
    torch.save(model.state_dict(), name)

    return name


def inference(cfg, name, test_loader, criterion):
    model = MLP(cfg)
    model.load_state_dict(torch.load(name))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avg_loss = [0] * len(cfg.data.digits)
    count = [0] * len(cfg.data.digits)
    y_classes = dict(zip(cfg.data.digits, range(len(cfg.data.digits))))

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = model(x.to(device))
            batch_onehot = y.apply_(lambda i: y_classes[i])
            loss = criterion(out, F.one_hot(batch_onehot, cfg.data.n_classes).to(torch.float))
            avg_loss[y] += loss.item()
            count[y] += 1

            if batch_idx == 0 and cfg.train.plot_sample:
                # Plot the distribution of the output
                probs = torch.softmax(out[0], dim=-1)
                plt.bar(np.arange(len(cfg.data.digits)), probs.cpu().numpy())
                plt.xlabel("Number of classes")
                plt.ylabel("Class probabilities (for y={})".format(y))
                plt.xticks(np.arange(len(cfg.data.digits)))
                plt.show()
            
    avg_loss = [avg_loss[i] / count[i] for i in range(len(cfg.data.digits))]

    if cfg.train.plot:
        plt.bar(list(y_classes.keys()), avg_loss)
        plt.xlabel("Number of classes")
        plt.ylabel("Average Test Loss")
        plt.xticks(list(y_classes.keys()))
        plt.show()

        print("")


@hydra.main(config_path="./configurations", config_name="train.yaml")
def main(cfg):
    # wandb.init(project="test-project", entity="model-driven-models")
    # wandb.config = cfg

    dataset = MNIST(cfg)
    train_loader, test_loader = dataset.create_dataloaders(unbalanced=cfg.data.unbalanced)
    model = MLP(cfg)
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum
    )
    criterion = torch.nn.CrossEntropyLoss()

    name = train(cfg, train_loader, test_loader, model, optimizer, criterion, unb_digits=cfg.data.unbalanced)
    
    test_loader = dataset.create_inference_dataloader()
    inference(cfg, name, test_loader, criterion)

    if cfg.train.fisher:
        cfg.train.name = name
        compute_fisher(cfg)

if __name__ == "__main__":
    main()
