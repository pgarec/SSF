import torch
import torch.nn.functional as F
import torch.optim as optim
import hydra
import pickle

from model_merging.model import MLP
from model_merging.data import MNIST
from model_merging import fisher
from model_merging.data import store_fisher


@hydra.main(config_path="./configurations", config_name="compute_fisher.yaml")
def main(cfg):
    model_name = cfg.data.model_name
    model = MLP(cfg)
    model.load_state_dict(torch.load('./models/'+model_name))

    dataset = MNIST(cfg)
    train_loader, _ = dataset.create_dataloaders()

    print("Starting Fisher computation")
    fisher_diag = fisher.compute_fisher_for_model(model, train_loader)
    print("Fisher computed. Saving to file...")
    fisher_name = "mnist_{}".format(''.join(map(str, cfg.data.digits)))
    store_fisher(fisher_diag, cfg.data.fisher_path+fisher_name)
    print("Fisher saved to file")

if __name__ == "__main__":
    main()


