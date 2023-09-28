"""
Functions borrowed from https://github.com/tudor-berariu/fisher-information-matrix/blob/master/fim.py
"""

import torch
import torch.nn as nn
import hydra

from .model import MLP
from .data import store_file
from .data import create_dataset
import random
import numpy as np

import time
import sys
from typing import Dict
from argparse import Namespace

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from torch.utils.data import DataLoader

sf = 1

def compute_fisher_model(model, dataset, num_classes, fisher_samples=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def fisher_single_example(single_example_batch):
        logits = model(single_example_batch.to(device))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sq_grads = []

        # Average of squared gradients of log probabilities respect to parameters for each class
        for i in range(num_classes):
            model.zero_grad()
            log_prob = log_probs[0][i]
            log_prob.backward(retain_graph=True)
            grad = [p.grad.clone() for p in model.parameters()]
            sq_grad = [probs[0][i] * g**2 for g in grad]
            sq_grads.append(sq_grad)
        
        log_prob.backward()
        model.zero_grad()

        return [torch.sum(torch.stack(g), dim=0)*sf for g in zip(*sq_grads)]

    variables = [p for p in model.parameters()]
    fishers = [torch.zeros(w.shape, requires_grad=False).to(device) for w in variables]

    n_examples = 0
    
    for batch, _ in dataset:
        n_examples += batch.shape[0]
        fishers_batch = torch.zeros((len(variables)),requires_grad=False).to(device)
        for element in batch:
            fish_elem = fisher_single_example(element.unsqueeze(0))
            fish_elem = [x.detach() for x in fish_elem]
            fishers_batch = [x + y for (x,y) in zip(fishers_batch, fish_elem)]

        # Sum of squared gradients
        fishers = [x+y for (x,y) in zip(fishers, fishers_batch)]

        if fisher_samples != -1 and n_examples > fisher_samples:
            break
     
    # Average over samples
    fishers = [fisher/n_examples for fisher in fishers]

    return fishers


def compute_empirical_gradients_model(model, dataset, num_classes, grad_samples=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
        
    def gradients_single_example(single_example_batch_x, single_example_batch_y):
        logits = model(single_example_batch_x.to(device))
        criterion = nn.CrossEntropyLoss(reduction='sum')
        loss = criterion(logits[0], single_example_batch_y[0])

        model.zero_grad()
        loss.backward()
        grad = [p.grad.clone() for p in model.parameters()]
            
        return grad

    variables = [p for p in model.parameters()]
    grads = [torch.zeros(w.shape, requires_grad=False).to(device) for w in variables]
    
    n_examples = 0
    for batch_x, batch_y in dataset:
        n_examples += batch_x.shape[0]
        grads_batch = torch.zeros((len(variables)),requires_grad=False).to(device)

        for element_x, element_y in zip(batch_x, batch_y):
            model.zero_grad()
            grad_elem = gradients_single_example(element_x.unsqueeze(0), element_y.unsqueeze(0))
            grad_elem = [x.detach() for x in grad_elem]
            grads_batch = [x + y for (x,y) in zip(grads_batch, grad_elem)]

        grads = [x+y for (x,y) in zip(grads, grads_batch)]

        if grad_samples != -1 and n_examples > grad_samples:
            break
    
    return grads


def compute_gradients_model(model, dataset, num_classes, grad_samples=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def gradients_single_example(single_example_batch):
        logits = model(single_example_batch.to(device))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        grads_classes = []

        for i in range(num_classes):
            model.zero_grad()
            log_prob = log_probs[0][i]
            log_prob.backward(retain_graph=True)
            grad = [p.grad.clone() for p in model.parameters()]
            g = [probs[0][i] * g  for g in grad]
            grads_classes.append(g)

        return [torch.sum(torch.stack(g), dim=0)*sf for g in zip(*grads_classes)]

    variables = [p for p in model.parameters()]
    grads = [torch.zeros(w.shape, requires_grad=False).to(device) for w in variables]
    
    n_examples = 0
    for batch_x, batch_y in dataset:
        n_examples += batch_x.shape[0]
        grads_batch = torch.zeros((len(variables)),requires_grad=False).to(device)

        for element in batch_x:
            model.zero_grad()
            grad_elem = gradients_single_example(element.unsqueeze(0))
            grad_elem = [x.detach() for x in grad_elem]
            grads_batch = [x + y for (x,y) in zip(grads_batch, grad_elem)]

        grads = [x+y for (x,y) in zip(grads, grads_batch)]

        if grad_samples != -1 and n_examples > grad_samples:
            break
    
    return grads


def fim_diag(model: Module,
             data_loader: DataLoader,
             samples_no: int = None,
             empirical: bool = False,
             device: torch.device = None,
             verbose: bool = False,
             every_n: int = None) -> Dict[int, Dict[str, Tensor]]:
    fim = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    seen_no = 0
    last = 0
    tic = time.time()

    all_fims = dict({})

    while samples_no is None or seen_no < samples_no:
        data_iterator = iter(data_loader)
        try:
            data, target = next(data_iterator)
        except StopIteration:
            if samples_no is None:
                break
            data_iterator = iter(data_loader)
            data, target = next(data_loader)

        if device is not None:
            data = data.to(device)
            if empirical:
                target = target.to(device)

        logits = model(data)
        if empirical:
            outdx = target.unsqueeze(1)
        else:
            outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
        samples = logits.gather(1, outdx)

        idx, batch_size = 0, data.size(0)
        while idx < batch_size and (samples_no is None or seen_no < samples_no):
            model.zero_grad()
            torch.autograd.backward(samples[idx], retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fim[name] += (param.grad * param.grad)
                    fim[name].detach_()
            seen_no += 1
            idx += 1

            if verbose and seen_no % 100 == 0:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
                tic, last = toc, seen_no
                sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

            if every_n and seen_no % every_n == 0:
                all_fims[seen_no] = {n: f.clone().div_(seen_no).detach_()
                                     for (n, f) in fim.items()}

    if verbose:
        if seen_no > last:
            toc = time.time()
            fps = float(seen_no - last) / (toc - tic)
        sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

    # for name, grad2 in fim.items():
    #     grad2 /= float(seen_no)

    all_fims[seen_no] = fim

    return [x for x in fim.values()]


def grad_diag(model: Module,
             data_loader: DataLoader,
             samples_no: int = None,
             empirical: bool = False,
             device: torch.device = None,
             verbose: bool = False,
             every_n: int = None) -> Dict[int, Dict[str, Tensor]]:
    fim = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    seen_no = 0
    last = 0
    tic = time.time()

    all_fims = dict({})

    while samples_no is None or seen_no < samples_no:
        data_iterator = iter(data_loader)
        try:
            data, target = next(data_iterator)
        except StopIteration:
            if samples_no is None:
                break
            data_iterator = iter(data_loader)
            data, target = next(data_loader)

        if device is not None:
            data = data.to(device)
            if empirical:
                target = target.to(device)

        logits = model(data)
        if empirical:
            outdx = target.unsqueeze(1)
        else:
            outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
        samples = logits.gather(1, outdx)

        idx, batch_size = 0, data.size(0)
        while idx < batch_size and (samples_no is None or seen_no < samples_no):
            model.zero_grad()
            torch.autograd.backward(samples[idx], retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fim[name] += (param.grad)
                    fim[name].detach_()
            seen_no += 1
            idx += 1

            if verbose and seen_no % 100 == 0:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
                tic, last = toc, seen_no
                sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

            if every_n and seen_no % every_n == 0:
                all_fims[seen_no] = {n: f.clone().div_(seen_no).detach_()
                                     for (n, f) in fim.items()}

    if verbose:
        if seen_no > last:
            toc = time.time()
            fps = float(seen_no - last) / (toc - tic)
        sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

    # for name, grad2 in fim.items():
    #     grad2 /= float(seen_no)

    all_fims[seen_no] = fim

    return [x for x in fim.values()]


def compute_fisher_diagonals(model, train_loader, num_classes, fisher_samples=10000):
    #fishers = compute_fisher_model(model, train_loader, num_classes, fisher_samples=fisher_samples)
    fishers = fim_diag(model, train_loader, fisher_samples=fisher_samples)

    return fishers


def compute_gradients(model, train_loader, num_classes, grad_samples=10000):
    #grad = compute_gradients_model(model, train_loader, num_classes, grad_samples=grad_samples)
    grads = grad_diag(model, train_loader, grad_samples=grad_samples)
    
    return grads


def compute_and_store_fisher_diagonals(model, model_name, fisher_path, train_loader, num_classes, fisher_samples=10000):
    print("Starting Fisher computation")
    #fishers = compute_fisher_model(model, train_loader, num_classes, fisher_samples=fisher_samples)
    fishers = fim_diag(model, train_loader, samples_no=fisher_samples)
    print("Fisher computed. Saving to file...")
    fisher_name = model_name.split('/')[-1][:-3]
    store_file(fishers, fisher_path + fisher_name)
    print("Fisher saved to file")


def compute_and_store_gradients(model, model_name, grad_path, train_loader, num_classes, grad_samples=10000):
    print("Starting Grads computation")
    # grads = compute_gradients_model(model, train_loader, num_classes, grad_samples=grad_samples)
    grads = grad_diag(model, train_loader, samples_no=grad_samples)
    print("Grads computed. Saving to file...")
    grad_name = model_name.split('/')[-1][:-3]
    store_file(grads, grad_path + grad_name)
    print("Grads saved to file")


if __name__ == "__main__":
    compute_and_store_fisher_diagonals()
