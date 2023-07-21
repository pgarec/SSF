import torch
import hydra

from .model import MLP
from .data import store_file
from .data import create_dataset
import random
import numpy as np
from torch.distributions import Categorical


def compute_fisher_model(model, dataset, num_classes, fisher_samples=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fisher_single_example_sampling(single_example_batch):
        logits = model(single_example_batch.to(device))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        grads = []

        outdx = Categorical(logits=log_probs).sample().unsqueeze(1).detach()
        samples = log_probs.gather(1, outdx)
        samples.backward(retain_graph=True)
        grad = [p.grad.clone() for p in model.parameters()]
        sq_grad = [g**2 for g in grad]
        grads.append(sq_grad)
        model.zero_grad()

        return [torch.sum(torch.stack(g), dim=0) for g in zip(*grads)]

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

        return [torch.sum(torch.stack(g), dim=0) for g in zip(*sq_grads)]

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


def compute_gradients_model(model, dataset, num_classes, grad_samples=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def gradients_single_example_sampling(single_example_batch):
        logits = model(single_example_batch.to(device))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        outdx = Categorical(logits=log_probs).sample().unsqueeze(1).detach()
        samples = log_probs.gather(1, outdx)
        samples.backward(retain_graph=True)
        grad = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        samples.backward()

        return grad 
        
    def gradients_single_example(single_example_batch):
        logits = model(single_example_batch.to(device))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        grads = []

        for i in range(num_classes):
            model.zero_grad()
            log_prob = log_probs[0][i]
            log_prob.backward(retain_graph=True)
            grad = [p.grad.clone() for p in model.parameters()]
            g = [probs[0][i] * g for g in grad]
            grads.append(g)

        log_prob.backward()
        model.zero_grad()

        return [torch.sum(torch.stack(g), dim=0) for g in zip(*grads)]

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


def compute_fisher_diagonals(model, train_loader, num_classes, fisher_samples=10000):
    fisher_diag = compute_fisher_model(model, train_loader, num_classes, fisher_samples=fisher_samples)
    
    return fisher_diag


def compute_gradients(model, train_loader, num_classes, grad_samples=10000):
    grad_diag = compute_gradients_model(model, train_loader, num_classes, grad_samples=grad_samples)
    
    return grad_diag


def compute_and_store_fisher_diagonals(model, model_name, fisher_path, train_loader, num_classes, fisher_samples=10000):
    print("Starting Fisher computation")
    fisher_diag = compute_fisher_model(model, train_loader, num_classes, fisher_samples=fisher_samples)
    print("Fisher computed. Saving to file...")
    fisher_name = model_name.split('/')[-1][:-3]
    store_file(fisher_diag, fisher_path + fisher_name)
    print("Fisher saved to file")


def compute_and_store_gradients(model, model_name, grad_path, train_loader, num_classes, fisher_samples=10000):
    print("Starting Grads computation")
    grad_diag = compute_gradients_model(model, train_loader, num_classes, grad_samples=fisher_samples)
    print("Grads computed. Saving to file...")
    grad_name = model_name.split('/')[-1][:-3]
    store_file(grad_diag, grad_path + grad_name)
    print("Grads saved to file")


if __name__ == "__main__":
    compute_and_store_fisher_diagonals()
