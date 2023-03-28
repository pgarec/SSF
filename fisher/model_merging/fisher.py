import torch
import hydra

from .model import MLP
from .data import store_file
from .data import create_dataset
import random


def _compute_exact_fisher_for_batch(batch, model, variables, num_classes):
    def fisher_single_example(single_example_batch):
        logits = model(single_example_batch)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sq_grads = []

        for i in range(num_classes):
            model.zero_grad()
            log_prob = log_probs[0][i]
            log_prob.backward(retain_graph=True)
            grad = [p.grad.clone() for p in model.parameters()]
            sq_grad = [probs[0][i] * g**2 for g in grad]
            sq_grads.append(sq_grad)
        
        log_prob.backward()
        
        return [torch.sum(torch.stack(g), dim=0) / num_classes for g in zip(*sq_grads)]

    fishers = torch.zeros((len(variables)),requires_grad=False)
    for element in batch:
       model.zero_grad()
       fish_elem = fisher_single_example(element.unsqueeze(0))
       fish_elem = [x.detach() for x in fish_elem]
       fishers = [x + y for (x,y) in zip(fishers, fish_elem)]
    
    return fishers


def compute_fisher_for_model(model, dataset, num_classes, fisher_samples=-1):
    variables = [p for p in model.parameters()]
    fishers = [torch.zeros(w.shape, requires_grad=False) for w in variables]

    n_examples = 0
    
    for batch, _ in dataset:
        n_examples += batch.shape[0]
        batch_fishers = _compute_exact_fisher_for_batch(
            batch, model, variables, num_classes
        )
        fishers = [x+y for (x,y) in zip(fishers, batch_fishers)]

        if fisher_samples != -1 and n_examples > fisher_samples:
            break

    for i, fisher in enumerate(fishers):
        fishers[i] = fisher / n_examples

    return fishers


def _compute_exact_grads_for_batch(batch, model, variables, num_classes):
    def grads_single_example(single_example_batch):
        logits = model(single_example_batch)
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

        return [torch.sum(torch.stack(g), dim=0) / num_classes for g in zip(*grads)]

    grads = torch.zeros((len(variables)),requires_grad=False)
    for element in batch:
       model.zero_grad()
       grad_elem = grads_single_example(element.unsqueeze(0))
       grad_elem = [x.detach() for x in grad_elem]
       grads = [x + y for (x,y) in zip(grads, grad_elem)]

    return grads


def compute_grads_for_model(model, dataset, num_classes, grad_samples=-1):
    variables = [p for p in model.parameters()]
    # list of the model variables initialized to zero
    grads = [torch.zeros(w.shape, requires_grad=False) for w in variables]

    n_examples = 0

    for batch, _ in dataset:
        n_examples += batch.shape[0]
        batch_grads = _compute_exact_grads_for_batch(
            batch, model, variables, num_classes
        )
        grads = [x+y for (x,y) in zip(grads, batch_grads)]

        if grad_samples != -1 and n_examples > grad_samples:
            break

    for i, grad in enumerate(grads):
        grads[i] = grad / n_examples

    return grads


def compute_fisher_diags_init(model, train_loader, num_classes, fisher_samples=10000):
    fisher_diag = compute_fisher_for_model(model, train_loader, num_classes, fisher_samples=fisher_samples)
    
    return fisher_diag


def compute_grads_init(model, train_loader, num_classes, grad_samples=10000):
    grad_diag = compute_grads_for_model(model, train_loader, num_classes, grad_samples=grad_samples)
    
    return grad_diag


def compute_fisher_diags(model, model_name, fisher_path, train_loader, num_classes, fisher_samples=10000):
    print("Starting Fisher computation")
    fisher_diag = compute_fisher_for_model(model, train_loader, num_classes, fisher_samples=fisher_samples)
    print("Fisher computed. Saving to file...")
    fisher_name = model_name.split('/')[-1][:-3]
    store_file(fisher_diag, fisher_path + fisher_name)
    print("Fisher saved to file")


def compute_fisher_grads(model, model_name, grad_path, train_loader, num_classes, fisher_samples=10000):
    print("Starting Grads computation")
    grad_diag = compute_grads_for_model(model, train_loader, num_classes, grad_samples=fisher_samples)
    print("Grads computed. Saving to file...")
    grad_name = model_name.split('/')[-1][:-3]
    store_file(grad_diag, grad_path + grad_name)
    print("Grads saved to file")


if __name__ == "__main__":
    compute_fisher_diags()
