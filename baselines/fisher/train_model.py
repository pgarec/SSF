import torch
import torch.nn.functional as F
import torch.optim as optim
import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb

from model_merging.data import MNIST   
from model_merging.model import MLP


def train(cfg, train_loader, test_loader, model, optimizer, criterion): 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x.to(device))
       
            loss = criterion(out, F.one_hot(y, cfg.data.n_classes).to(torch.float))
            loss.backward()
            optimizer.step()
            train_loss += loss
            scheduler.step()
            # wandb.log({"Training loss": loss/len(train_loader)})
        
        print(f"Epoch [{epoch + 1}/{cfg.train.epochs}], Training Loss: {train_loss/len(train_loader):.4f}")

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for batch_idx, (x, y) in enumerate(test_loader):
                out = model(x.to(device))
                loss = criterion(out, F.one_hot(y, cfg.data.n_classes).to(torch.float))
                val_loss += loss
            # wandb.log({"Validation loss": loss/len(val_loader)})
            print(f"Epoch [{epoch + 1}/{cfg.train.epochs}], Validation Loss: {val_loss/len(test_loader):.4f}")    
    
    print('')
    name = './models/mnist_{}_epoch{}.pt'.format(''.join(map(str, cfg.data.digits)), epoch)
    torch.save(model.state_dict(), name)

    return name 


def inference(cfg, name, test_loader):
    model = MLP(cfg)
    model.load_state_dict(torch.load(name))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = model(x.to(device))

            # Plot the distribution of the output
            probs = torch.softmax(out[0], dim=-1)
            print('The target number is {}'.format(y[0]))
            print('The predicted number is {}'.format(torch.argmax(probs)))

            plt.bar(np.arange(len(cfg.data.digits)), probs.cpu().numpy())
            plt.xlabel("Number of classes")
            plt.ylabel("Class probabilities (for y={})".format(y[0]))
            plt.xticks(np.arange(len(cfg.data.digits)))
            plt.show()

            break


@hydra.main(config_path="./configurations", config_name="data.yaml")
def main(cfg):
    # wandb.init(project="test-project", entity="model-driven-models")
    # wandb.config = cfg

    dataset = MNIST(cfg)
    train_loader, test_loader = dataset.create_dataloaders()
    model = MLP(cfg)
    optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)
    criterion = torch.nn.CrossEntropyLoss()

    name = train(cfg, train_loader, test_loader, model, optimizer, criterion)
    inference(cfg, name, test_loader)


if __name__ == "__main__":
    main()