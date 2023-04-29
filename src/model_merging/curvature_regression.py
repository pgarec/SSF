import torch
import hydra

from .model import MLP_regression
from .data import store_file
from .data import create_dataset
import torch
import math


def log_prob(logit, y, sigma_sq):
    sigma_sq = 0.01
    sigma_sq = torch.tensor([sigma_sq])
    logprob = - 0.5*torch.log(2*torch.tensor([math.pi])) - 0.5*torch.log(sigma_sq) - ((0.5*(1/sigma_sq)*(logit-y)**2))

    return logprob


def compute_fisher_model(model, dataset, fisher_samples=-1, sigma_sq=-1):
    def fisher_single_example(x, y, sigma_sq):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logit = model(x).to(device)
        lp = log_prob(logit, y, sigma_sq)
        lp.backward()
        grad = [p.grad.clone() for p in model.parameters()]
        sq_grad = [g**2 for g in grad]

        return sq_grad
    
    variables = [p for p in model.parameters()]
    fishers = [torch.zeros(w.shape, requires_grad=False) for w in variables]
    n_examples = 0

    for batch_x, batch_y in dataset:
        print(n_examples)
        n_examples += batch_x.shape[0]
        
        fishers_batch = torch.zeros((len(variables)),requires_grad=False)
        for element_x, element_y in zip(batch_x, batch_y):
            model.zero_grad()
            fish_elem = fisher_single_example(element_x.unsqueeze(0), element_y.unsqueeze(0), sigma_sq)
            fish_elem = [x.detach() for x in fish_elem]
            fishers_batch = [x + y for (x,y) in zip(fishers_batch, fish_elem)]

        fishers = [x+y for (x,y) in zip(fishers, fishers_batch)]

        if fisher_samples != -1 and n_examples > fisher_samples:
            break
        
    for i, fisher in enumerate(fishers):
        fishers[i] = fisher / n_examples

    return fishers


def compute_gradients_model(model, dataset, grad_samples=-1, sigma_sq=-1):
    def grads_single_example(x, y, sigma_sq):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.zero_grad()
        logit = model(x).to(device)
        lp = log_prob(logit, y, sigma_sq)
        lp.backward()
        grad = [p.grad.clone() for p in model.get_trainable_parameters()]

        return grad

    variables = [p for p in model.parameters()]
    grads = [torch.zeros(w.shape, requires_grad=False) for w in variables]
    n_examples = 0

    for batch_x, batch_y in dataset:
        print(n_examples)
        n_examples += batch_x.shape[0]
        batch_grads = torch.zeros((len(variables)),requires_grad=False)

        for element_x, element_y in zip(batch_x, batch_y):
            grad_elem = grads_single_example(element_x.unsqueeze(0), element_y.unsqueeze(0), sigma_sq)
            grad_elem = [x.detach() for x in grad_elem]
            batch_grads = [x + y for (x,y) in zip(batch_grads, grad_elem)]

        grads = [x+y for (x,y) in zip(grads, batch_grads)]

        if grad_samples != -1 and n_examples > grad_samples:
            break
            
    # for i, grad in enumerate(grads):
    #     grads[i] = grad / n_examples

    return grads


def compute_gradients_model_mse(model, dataset, grad_samples=-1, sigma_sq=-1):
    def grads_single_example(x, y, sigma_sq):
        model.zero_grad()
        logit = model(x)
        loss = criterion(logit, y)
        loss.backward()
        grad = [p.grad.clone() for p in model.parameters()]

        return grad

    variables = [p for p in model.parameters()]
    grads = [torch.zeros(w.shape, requires_grad=False) for w in variables]
    n_examples = 0
    criterion = torch.nn.MSELoss()

    for batch_x, batch_y in dataset:
        print(n_examples)
        n_examples += batch_x.shape[0]
        batch_grads = torch.zeros((len(variables)),requires_grad=False)

        for element_x, element_y in zip(batch_x, batch_y):
            grad_elem = grads_single_example(element_x.unsqueeze(0), element_y.unsqueeze(0), sigma_sq)
            grad_elem = [x.detach() for x in grad_elem]
            batch_grads = [x + y for (x,y) in zip(batch_grads, grad_elem)]

        grads = [x+y for (x,y) in zip(grads, batch_grads)]

        if grad_samples != -1 and n_examples > grad_samples:
            break
        
    for i, grad in enumerate(grads):
        grads[i] = grad / n_examples

    return grads


def compute_and_store_fisher_diagonals(cfg, model_name, train_loader=[]):
    model = MLP_regression(cfg)
    model.eval()
    model.load_state_dict(torch.load(model_name))

    if train_loader == []:
        dataset = create_dataset(cfg)
        train_loader, _ = dataset.create_dataloaders()

    print("Starting Fisher computation")
    fisher_diag = compute_fisher_model(model, train_loader, fisher_samples=cfg.data.fisher_samples, sigma_sq=cfg.train.sigma_sq)
    print("Fisher computed. Saving to file...")
    fisher_name = model_name.split('/')[-1][:-3]
    store_file(fisher_diag, cfg.data.fisher_path + fisher_name)
    print("Fisher saved to file")


def compute_and_store_gradients(cfg, model_name, train_loader=[]):
    model = MLP_regression(cfg)
    model.eval()
    model.load_state_dict(torch.load(model_name))

    if train_loader == []:
        dataset = create_dataset(cfg)
        train_loader, _ = dataset.create_dataloaders()

    print("Starting Grads computation")
    grad_diag = compute_gradients_model(model, train_loader, grad_samples=1000, sigma_sq=cfg.train.sigma_sq)
    print("Grads computed. Saving to file...")
    grad_name = model_name.split('/')[-1][:-3]
    store_file(grad_diag, cfg.data.grad_path + grad_name)
    print("Grads saved to file")


if __name__ == "__main__":
    compute_and_store_fisher_diagonals()
