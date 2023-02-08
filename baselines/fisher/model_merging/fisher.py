import torch


def _compute_exact_fisher_for_batch(batch, model, variables, expectation_wrt_logits):
    num_classes = model.num_classes

    def fisher_single_example(single_example_batch):
        log_probs = model(single_example_batch, training=True)
        probs = torch.nn.functional.softmax(log_probs, dim=-1)
        # calculates the gradients of the log-probs with respect to the variables
        # (the parameters of the model), and squares them
        sq_grads = []
        for i in range(num_classes):
            log_prob = log_probs[0][i]
            grad = torch.autograd.grad(log_prob, variables, retain_graph=True)

            # square of gradients == hessian
            sq_grad = [probs[0][i] * g**2 for g in grad]
            sq_grads.append(sq_grad)

        # The squared gradients are then summed over all classes, resulting in a list of scalars,
        # which represents the Fisher information matrix for the single example.

        example_fisher = [torch.sum(torch.stack(g), dim=0) for g in zip(*sq_grads)]

        return example_fisher

    # computes the Fisher information matrix for each example in the batch
    fishers = [
        fisher_single_example(batch[i].unsqueeze(0)) for i in range(batch.size(0))
    ]

    # function returns the sum of the Fisher information matrix across all examples in the batch.
    # This result is a list of scalars, one for each variable (parameter) in the model.

    return [torch.sum(torch.stack(g), dim=0) for g in zip(*fishers)]


def compute_fisher_for_model(model, dataset, expectation_wrt_logits=True):
    variables = [p for p in model.parameters()]
    # list of the model variables initialized to zero
    fishers = [torch.zeros(w.shape, requires_grad=False) for w in variables]

    n_examples = 0
    for batch, _ in dataset:
        n_examples += batch.shape[0]
        batch_fishers = _compute_exact_fisher_for_batch(
            batch, model, variables, expectation_wrt_logits
        )
        for f, bf in zip(fishers, batch_fishers):
            f.add_(bf)

    for i, fisher in enumerate(fishers):
        fishers[i] = fisher / n_examples

    return fishers
