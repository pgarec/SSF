import torch
import hydra

from .model import MLP
from .data import store_file
from .data import create_dataset


def _compute_exact_fisher_for_batch(batch, model, variables, expectation_wrt_logits):
    num_classes = model.num_classes

    def fisher_single_example(single_example_batch):
        # calculates the gradients of the log-probs with respect to the variables
        # (the parameters of the model), and squares them
        logits = model(single_example_batch)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(log_probs, dim=-1)
        sq_grads = []

        for i in range(num_classes):
            log_prob = log_probs[0][i]
            log_prob.backward(retain_graph=True)
            grad = [p.grad.clone() for p in model.parameters()]
            sq_grad = [probs[0][i] * g**2 for g in grad]
            sq_grads.append(sq_grad)

        return [torch.sum(torch.stack(g), dim=0) / num_classes for g in zip(*sq_grads)]

    fishers = torch.zeros((len(variables)),requires_grad=False)
    for element in batch:
       model.zero_grad()
       fish_elem = fisher_single_example(element.unsqueeze(0))
       fish_elem = [x.detach() for x in fish_elem]
       fishers = [x + y for (x,y) in zip(fishers, fish_elem)]

    return fishers


def compute_fisher_for_model(model, dataset, expectation_wrt_logits=True, fisher_samples=-1):
    variables = [p for p in model.parameters()]
    # list of the model variables initialized to zero
    fishers = [torch.zeros(w.shape, requires_grad=False) for w in variables]

    n_examples = 0

    for batch, _ in dataset:
        print(n_examples)
        n_examples += batch.shape[0]
        batch_fishers = _compute_exact_fisher_for_batch(
            batch, model, variables, expectation_wrt_logits
        )
        fishers = [x+y for (x,y) in zip(fishers, batch_fishers)]

        if fisher_samples != -1 and n_examples > fisher_samples:
            break

    for i, fisher in enumerate(fishers):
        fishers[i] = fisher / n_examples

    return fishers


def _compute_exact_grads_for_batch(batch, model, variables, expectation_wrt_logits):
    num_classes = model.num_classes

    def grads_single_example(single_example_batch):
        logits = model(single_example_batch)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(log_probs, dim=-1)
        sq_grads = []

        for i in range(num_classes):
            log_prob = log_probs[0][i]
            log_prob.backward(retain_graph=True)
            grad = [p.grad.clone() for p in model.parameters()]
            sq_grad = [probs[0][i] * g for g in grad]
            sq_grads.append(sq_grad)


        return [torch.sum(torch.stack(g), dim=0) / num_classes for g in zip(*sq_grads)]

    grads = torch.zeros((len(variables)),requires_grad=False)
    for element in batch:
       model.zero_grad()
       grad_elem = grads_single_example(element.unsqueeze(0))
       grad_elem = [x.detach() for x in grad_elem]
       grads = [x + y for (x,y) in zip(grads, grad_elem)]

    return grads


def compute_grads_for_model(model, dataset, expectation_wrt_logits=True, grad_samples=-1):
    variables = [p for p in model.parameters()]
    # list of the model variables initialized to zero
    grads = [torch.zeros(w.shape, requires_grad=False) for w in variables]

    n_examples = 0

    for batch, _ in dataset:
        print(n_examples)
        n_examples += batch.shape[0]
        batch_grads = _compute_exact_grads_for_batch(
            batch, model, variables, expectation_wrt_logits
        )
        grads = [x+y for (x,y) in zip(grads, batch_grads)]

        if grad_samples != -1 and n_examples > grad_samples:
            break

    for i, grad in enumerate(grads):
        grads[i] = grad / n_examples

    return grads


def compute_fisher_diags(cfg, model_name):
    model = MLP(cfg)
    model.load_state_dict(torch.load(model_name))

    dataset = create_dataset(cfg)
    train_loader, _ = dataset.create_dataloaders(unbalanced=cfg.data.unbalanced)

    print("Starting Fisher computation")
    fisher_diag = compute_fisher_for_model(model, train_loader, fisher_samples=cfg.data.fisher_samples)
    print("Fisher computed. Saving to file...")
    fisher_name = model_name.split('/')[-1][:-3]
    store_file(fisher_diag, cfg.data.fisher_path + fisher_name)
    print("Fisher saved to file")


def compute_fisher_grads(cfg, model_name):
    model = MLP(cfg)
    model.load_state_dict(torch.load(model_name))

    dataset = create_dataset(cfg)
    train_loader, _ = dataset.create_dataloaders(unbalanced=cfg.data.unbalanced)

    print("Starting Grads computation")
    grad_diag = compute_grads_for_model(model, train_loader, grad_samples=cfg.data.fisher_samples)
    print("Grads computed. Saving to file...")
    grad_name = model_name.split('/')[-1][:-3]
    store_file(grad_diag, cfg.data.grad_path + grad_name)
    print("Grads saved to file")


if __name__ == "__main__":
    compute_fisher_diags()
