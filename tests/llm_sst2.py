'''
File with the experiment LLMs.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import datasets as hfds

import matplotlib.colors as colors
import seaborn as sns
from torch import nn
# from manifold import cross_entropy_manifold
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from src.model_merging.datasets.pinwheel import make_pinwheel_data
import hydra
from src.model_merging.data import load_fishers, load_grads
from src.model_merging.curvature import fim_diag_llm, grad_diag_llm, compute_and_store_gradients, compute_and_store_fisher_diagonals
from src.model_merging.merging import merging_models_fisher, merging_models_isotropic
from src.model_merging.model import get_mergeable_variables
from src.merge_permutation import merging_models_permutation
import omegaconf

import torchtext
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import datasets as hfds


# CONFIGURATION
cfg = omegaconf.OmegaConf.load('./configurations/perm_llm_sst2.yaml')
sequence_length = 128
seed = cfg.train.torch_seed

if seed > -1:
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def clone_model(model, num_features, H, num_output, seed):
    cloned = model.__class__(num_features, H, num_output, seed)
    cloned.load_state_dict(model.state_dict().copy())
        
    return cloned   


def load_metric_for_glue_task(task: str):
    return hfds.load_metric("glue", task)


def evaluate_model(model, dataset, metric):
    for input in dataset:
        input_ids = torch.stack(input["input_ids"], dim=-1)
        attention_mask = torch.stack(input["attention_mask"], dim=-1)
        model_predictions = model(input_ids, attention_mask=attention_mask).logits
        model_predictions = torch.argmax(model_predictions, axis=-1)
        metric.add_batch(predictions=model_predictions, references=input["label"])
    return metric.compute()


def average_score(score):
    return sum(score.values()) / len(score.values())


def encode(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length")


if __name__ == "__main__":  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    models.append(AutoModelForSequenceClassification.from_pretrained("aviator-neural/bert-base-uncased-sst2"))
    models.append(AutoModelForSequenceClassification.from_pretrained("howey/bert-base-uncased-sst2"))
    models.append(AutoModelForSequenceClassification.from_pretrained("yoshitomo-matsubara/bert-base-uncased-sst2"))

    criterion = hfds.load_metric("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset, val_dataset = load_dataset("glue", "sst2", split=['train','validation[:100]'])
    train_dataset = train_dataset.map(encode, batched=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    val_dataset = val_dataset.map(encode, batched=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    parameters = get_mergeable_variables(models[0])
    metatheta = nn.utils.parameters_to_vector(parameters)
    print("Number of parameters: {}".format(len(metatheta)))

    names = []
    for n, model in enumerate(models):
        name = "{}{}_model{}.pt".format(
                cfg.data.model_path,
                cfg.data.dataset,
                n)
        names.append(name)
        compute_and_store_fisher_diagonals(model, name, cfg.data.fisher_path, val_loader, fisher_samples=cfg.data.grad_samples, llm=True)
        compute_and_store_gradients(model, name, cfg.data.grad_path, val_loader, grad_samples=cfg.data.grad_samples, llm=True)

    grads = load_grads(cfg, names)
    fishers = load_fishers(cfg, names)

    # for n, model in enumerate(models):
    #     print("Loss model {}:{}".format(n, evaluate_model(model, val_loader, criterion)))

    output_model = models[0]
    isotropic_model = merging_models_isotropic(output_model, models)
    print("Istropic model loss: {}".format(evaluate_model(isotropic_model, val_loader, criterion)))

    output_model = models[0]
    fishers = [fim_diag_llm(m, val_loader, cfg.data.n_examples) for m in models]
    fisher_model = merging_models_fisher(output_model, models, fishers)
    print("Fisher model loss: {}".format(evaluate_model(fisher_model, val_loader, criterion)))

    grads = [grad_diag_llm(m, train_loader, cfg.data.n_examples) for m in models]
    cfg.train.initialization = "MLP"
    # output_model = clone_model(models[0], num_features, H, num_output, seed)
    output_model = models[0]
    config = AutoConfig.from_pretrained("bert-base-uncased")
    metamodel = AutoModelForSequenceClassification.from_config(config)
    perm_model, _, _ = merging_models_permutation(cfg, metamodel, models, grads, fishers, test_loader=val_loader, llm=True, criterion=criterion, plot=False, store=False)
    print("Permutation model loss: {}".format(evaluate_model(perm_model, val_loader, criterion)))

