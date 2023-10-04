# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CROMAI  ---  (pol.garcia@bsc.es)
# Barcelona Supercomputing Center (BSC)

import torch
import hydra
from model_merging.data import load_models, store_file, create_dataset
from .merge_fisher import evaluate_fisher
from .merge_permutation import evaluate_permutation
from .merge_isotropic import evaluate_isotropic
from .train import inference
from model_merging.model import MLP
from model_merging.merging import merging_models_isotropic
from model_merging.evaluation import plot_avg_merging_techniques, plot_merging_techniques
import torch.nn.functional as F


def store_results(cfg, isotropic_loss, fisher_loss, output_loss):
    d = "".join(map(str, cfg.data.classes))
    store_file(isotropic_loss, cfg.data.results_path + "isotropic_loss_{}".format(d))
    store_file(fisher_loss, cfg.data.results_path + "fisher_loss_{}".format(d))
    store_file(output_loss, cfg.data.results_path + "output_loss_{}".format(d))


def evaluate_techniques(cfg, model_names = []):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_names == []:
        models = load_models(cfg)
    else:
        models = load_models(cfg, model_names)

    dataset = create_dataset(cfg)
    test_loader = dataset.create_inference_dataloader()
    
    y_classes = dict(zip(cfg.data.classes, range(len(cfg.data.classes))))
    criterion = torch.nn.CrossEntropyLoss()
    outputs = []

    for model in models:
        loss = inference(cfg, model, test_loader, criterion)
        print("loss model {}".format(loss))

    isotropic_loss, count = evaluate_isotropic(cfg, models, test_loader, criterion)
    fisher_loss, count = evaluate_fisher(cfg, models, test_loader, criterion, model_names)

    metamodel = MLP(cfg)
    metamodel = merging_models_isotropic(cfg, models)
    cfg.train.epochs = 500
    perm_loss, count = evaluate_permutation(cfg, metamodel, models, test_loader, criterion, model_names)

    output_loss = [0] * len(cfg.data.classes)
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
    results = {'Isotropic loss': sum(isotropic_loss)/len(test_loader), 'Fisher loss': sum(fisher_loss)/len(test_loader), 'Perm loss': sum(perm_loss)/len(test_loader), 'output_loss': loss}
    plot_avg_merging_techniques(results)
    print(results)
    
    isotropic_loss_avg = [isotropic_loss[i] / count[i] for i in range(len(cfg.data.classes))]
    fisher_loss_avg = [fisher_loss[i] / count[i] for i in range(len(cfg.data.classes))]
    perm_loss_avg = [perm_loss[i] / count[i] for i in range(len(cfg.data.classes))]
    output_loss_avg = [output_loss[i] / count[i] for i in range(len(cfg.data.classes))]
    plot_merging_techniques(cfg, isotropic_loss_avg, fisher_loss_avg, perm_loss_avg, output_loss_avg)

    store_results(cfg, isotropic_loss, fisher_loss, output_loss)


@hydra.main(config_path="./configurations", config_name="merge.yaml")
def main(cfg):
    evaluate_techniques(cfg, ["model_0.pt", "model_1.pt", "model_2.pt"])


if __name__ == "__main__":
    main()
