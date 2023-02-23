import torch
import hydra

from model_merging.model import MLP
from model_merging import fisher
from model_merging.data import store_file
from model_merging.data import create_dataset


@hydra.main(config_path="./configurations", config_name="compute_fisher.yaml")
def compute_fisher_diags(cfg):
    model_name = cfg.train.name
    model = MLP(cfg)
    model.load_state_dict(torch.load(model_name))

    dataset = create_dataset(cfg)
    train_loader, _ = dataset.create_dataloaders(unbalanced=cfg.data.unbalanced)

    print("Starting Fisher computation")
    fisher_diag = fisher.compute_fisher_for_model(model, train_loader, fisher_samples=cfg.data.fisher_samples)
    print("Fisher computed. Saving to file...")
    if cfg.data.unbalanced:
        d = "".join(map(str, cfg.data.classes))
        u = "".join(map(str, cfg.data.unbalanced))
        fisher_name = "mnist_{}_epoch{}_{}".format(d, cfg.train.epochs, u)

    else:
        d = "".join(map(str, cfg.data.classes))
        fisher_name = "mnist_{}_epoch{}".format(d, cfg.train.epochs)
    store_file(fisher_diag, cfg.data.fisher_path + fisher_name)
    print("Fisher saved to file")


if __name__ == "__main__":
    main()
