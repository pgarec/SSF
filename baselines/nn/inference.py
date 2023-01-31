import torch
import sys 
import hydra
import omegaconf
from torch.nn import CrossEntropyLoss

sys.path.append("./utils")
sys.path.append("./models")
from data_utils import get_mnist_loaders
from network import Model

config_model = "basic.yaml"
config_path = "./configurations"


def load_data(path_data):
    return get_mnist_loaders(path_data)


def load_model(cfg, path_model, name_model, device):
    model = Model(cfg)
    model.load_state_dict(torch.load(path_model+name_model))

    return model.to(device)


def step(cfg, model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CrossEntropyLoss()

    with torch.no_grad():
        loss_val = 0
        for batch_idx, (x, y) in enumerate(loader):
            out = model(x.to(device))
            y = torch.nn.functional.one_hot(y, cfg.data.n_classes)
            loss = criterion(out, y.float().to(device))
            loss_val += loss
        
        print(f" Validation loss:{loss/len(loader):.4f}")


@hydra.main(config_path=config_path, config_name=config_model)
def inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_model = "./baselines/nn/models/"
    name_model = "model_epoch0.mod"
    path_data = "./data"

    model = load_model(cfg, path_model, name_model, device)
    train_loader, val_loader, test_loader = load_data(path_data)
    step(cfg, model, test_loader)


if __name__ == "__main__":
    inference()
