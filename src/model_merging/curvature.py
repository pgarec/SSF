# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CROMAI  ---  (pol.garcia@bsc.es)
# Barcelona Supercomputing Center (BSC)

"""
The code is adapted from https://github.com/tudor-berariu/fisher-information-matrix/blob/master/fim.py
"""

import torch
from .data import store_file
import time
import sys
from typing import Dict

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from torch.utils.data import DataLoader

sf = 1

def fim_diag_llm(model: Module,
             data_loader: DataLoader,
             samples_no: int = None,
             empirical: bool = False,
             device: torch.device = None,
             verbose: bool = True,
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
            input = next(data_iterator)
        except StopIteration:
            if samples_no is None:
                break
            data_iterator = iter(data_loader)
            input = next(data_loader)   

       
        data = torch.stack(input["input_ids"], dim=-1)
        att = torch.stack(input["attention_mask"], dim=-1)
        target = input["label"]

        if device is not None:
            data = data.to(device)
            att = att.to(device)
            if empirical:
                target = target.to(device)

        logits = model(data, attention_mask=att).logits
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

            if verbose and seen_no % 2 == 0:
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


def grad_diag_llm(model: Module,
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
            input = next(data_iterator)
        except StopIteration:
            if samples_no is None:
                break
            data_iterator = iter(data_loader)
            input = next(data_loader)   

        data = torch.stack(input["input_ids"], dim=-1)
        att = torch.stack(input["attention_mask"], dim=-1)
        target = input["label"]

        if device is not None:
            data = data.to(device)
            att = att.to(device)
            if empirical:
                target = target.to(device)

        logits = model(data, attention_mask=att).logits
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

    all_fims[seen_no] = fim

    return [x for x in fim.values()]


def compute_fisher_diagonals(model, train_loader, num_classes, fisher_samples=10000):
    fishers = fim_diag(model, train_loader, fisher_samples=fisher_samples)

    return fishers


def compute_gradients(model, train_loader, num_classes, grad_samples=10000):
    grads = grad_diag(model, train_loader, grad_samples=grad_samples)
    
    return grads


def compute_and_store_fisher_diagonals(model, model_name, fisher_path, train_loader, fisher_samples=10000, llm=False):
    if llm:
        print("Starting Fisher computation")
        fishers = fim_diag_llm(model, train_loader, samples_no=fisher_samples)
        print("Fisher computed. Saving to file...")
        fisher_name = model_name.split('/')[-1][:-3]
        store_file(fishers, fisher_path + fisher_name)
        print("Fisher saved to file")
    else:
        print("Starting Fisher computation")
        fishers = fim_diag(model, train_loader, samples_no=fisher_samples)
        print("Fisher computed. Saving to file...")
        fisher_name = model_name.split('/')[-1][:-3]
        store_file(fishers, fisher_path + fisher_name)
        print("Fisher saved to file")


def compute_and_store_gradients(model, model_name, grad_path, train_loader, grad_samples=10000, llm=False):
    if llm:
        print("Starting Grads computation")
        grads = grad_diag_llm(model, train_loader, samples_no=grad_samples)
        print("Grads computed. Saving to file...")
        grad_name = model_name.split('/')[-1][:-3]
        store_file(grads, grad_path + grad_name)
        print("Grads saved to file")
    else:
        print("Starting Grads computation")
        grads = grad_diag(model, train_loader, samples_no=grad_samples)
        print("Grads computed. Saving to file...")
        grad_name = model_name.split('/')[-1][:-3]
        store_file(grads, grad_path + grad_name)
        print("Grads saved to file")


if __name__ == "__main__":
    compute_and_store_fisher_diagonals()
