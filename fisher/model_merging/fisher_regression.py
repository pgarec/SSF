import torch
import hydra

from .model import MLP_regression
from .data import store_file
from .data import create_dataset


def _compute_exact_fisher_for_batch(batch, model, variables):
    def fisher_single_example(single_example_batch):
        logits = model(single_example_batch)
        log_logits = torch.log(logits[0])
        log_logits.backward()
        grad = [p.grad.clone() for p in model.parameters()]
        sq_grad = [g**2 for g in grad]

        return sq_grad

    fishers = torch.zeros((len(variables)),requires_grad=False)
    for element in batch:
       model.zero_grad()
       fish_elem = fisher_single_example(element.unsqueeze(0))
       fish_elem = [x.detach() for x in fish_elem]
       fishers = [x + y for (x,y) in zip(fishers, fish_elem)]

    return fishers


def compute_fisher_for_model(model, dataset, fisher_samples=-1):
    variables = [p for p in model.parameters()]
    # list of the model variables initialized to zero
    fishers = [torch.zeros(w.shape, requires_grad=False) for w in variables]

    n_examples = 0

    for batch, _ in dataset:
        print(n_examples)
        n_examples += batch.shape[0]
        batch_fishers = _compute_exact_fisher_for_batch(
            batch, model, variables
        )
        fishers = [x+y for (x,y) in zip(fishers, batch_fishers)]

        if fisher_samples != -1 and n_examples > fisher_samples:
            break

    for i, fisher in enumerate(fishers):
        fishers[i] = fisher / n_examples

    return fishers


def _compute_exact_grads_for_batch(batch, model, variables):

    def grads_single_example(single_example_batch):
        logits = model(single_example_batch)
        log_logits = torch.log(logits[0])
        log_logits.backward()
        grad = [p.grad.clone() for p in model.parameters()]

        return grad

    grads = torch.zeros((len(variables)),requires_grad=False)

    for element in batch:
       model.zero_grad()
       grad_elem = grads_single_example(element.unsqueeze(0))
       grad_elem = [x.detach() for x in grad_elem]
       grads = [x + y for (x,y) in zip(grads, grad_elem)]

    return grads


def compute_grads_for_model(model, dataset, grad_samples=-1):
    variables = [p for p in model.parameters()]
    # list of the model variables initialized to zero
    grads = [torch.zeros(w.shape, requires_grad=False) for w in variables]

    n_examples = 0

    for batch, _ in dataset:
        print(n_examples)
        n_examples += batch.shape[0]
        batch_grads = _compute_exact_grads_for_batch(
            batch, model, variables
        )
        grads = [x+y for (x,y) in zip(grads, batch_grads)]

        if grad_samples != -1 and n_examples > grad_samples:
            break

    for i, grad in enumerate(grads):
        grads[i] = grad / n_examples

    return grads


def compute_fisher_diags(cfg, model_name, train_loader=[]):
    model = MLP_regression(cfg)
    model.eval()
    model.load_state_dict(torch.load(model_name))

    if train_loader == []:
        dataset = create_dataset(cfg)
        train_loader, _ = dataset.create_dataloaders()

    print("Starting Fisher computation")
    fisher_diag = compute_fisher_for_model(model, train_loader, fisher_samples=cfg.data.fisher_samples)
    print("Fisher computed. Saving to file...")
    fisher_name = model_name.split('/')[-1][:-3]
    store_file(fisher_diag, cfg.data.fisher_path + fisher_name)
    print("Fisher saved to file")


def compute_fisher_grads(cfg, model_name, train_loader=[]):
    model = MLP_regression(cfg)
    model.eval()
    model.load_state_dict(torch.load(model_name))

    if train_loader == []:
        dataset = create_dataset(cfg)
        train_loader, _ = dataset.create_dataloaders()

    print("Starting Grads computation")
    grad_diag = compute_grads_for_model(model, train_loader, grad_samples=cfg.data.fisher_samples)
    print("Grads computed. Saving to file...")
    grad_name = model_name.split('/')[-1][:-3]
    store_file(grad_diag, cfg.data.grad_path + grad_name)
    print("Grads saved to file")


if __name__ == "__main__":
    compute_fisher_diags()
