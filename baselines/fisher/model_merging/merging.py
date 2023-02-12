import collections
import torch
import torch.nn as nn
import torch.distributions as dist

from .model import clone_model, get_mergeable_variables

MergeResult = collections.namedtuple("MergeResult", ["coefficients", "score"])


def print_merge_result(result):
    print(f"Merging coefficients: {result.coefficients}")
    print("Scores:")
    for name, value in result.score.items():
        print(f"  {name}: {value}")


def create_pairwise_grid_coeffs(n_weightings):
    n_weightings -= 2
    denom = n_weightings + 1
    weightings = [((i + 1) / denom, 1 - (i + 1) / denom) for i in range(n_weightings)]
    weightings = [(0.0, 1.0)] + weightings + [(1.0, 0.0)]
    weightings.reverse()

    return weightings


def create_random_coeffs(n_models, n_weightings, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    dist = dist.Dirichlet(torch.ones([n_models]))
    return dist.sample((n_weightings,)).mean(dim=0).tolist()


def _merge_with_coeffs(
    output_variables,
    variables_to_merge,
    coefficients,
    fishers=None,
    fisher_floor=1e-6,
    favor_target_model=True,
    normalization_constants=None,
):
    
    n_models = len(variables_to_merge)
    assert len(coefficients) == n_models

    if fishers is None:
        fishers = n_models * [1.0]
    else:
        assert len(fishers) == n_models

    if normalization_constants is not None:
        assert len(normalization_constants) == n_models
        coefficients = [w / n for w, n in zip(coefficients, normalization_constants)]

    for i, var in enumerate(output_variables):
        lhs, rhs = [], []
        for j, (mvars, coeff, fisher) in enumerate(
            zip(variables_to_merge, coefficients, fishers)
        ):
            diag = fisher if isinstance(fisher, float) else fisher[i]
            if not favor_target_model or j == 0:
                # ensure that fisher diagonal doesn't vanish
                diag = torch.clamp(diag, min=fisher_floor, max=float("inf"))
            mvar = mvars[i]

            tmp = coeff * diag
            # coeff * diag
            lhs.append(tmp)
            # coeff * diag * parameter
            rhs.append(tmp * mvar.data)
        rhs = torch.stack(rhs).sum(dim=0)
        lhs = sum(lhs)
        # closed-form form solution of argmax
        var.data = rhs / lhs


def _l2_norm_of_fisher(fisher):
    norm_const = sum([torch.sum(d**2) for d in fisher]).item()

    return torch.sqrt(torch.tensor(norm_const))


def generate_merged_for_coeffs_set(
    cfg,
    mergeable_models,
    coefficients_set,
    fishers=None,
    fisher_floor=1e-6,
    favor_target_model=True,
    normalize_fishers=True,
):
    # Create the model to yield, then handle the norm_constants
    if normalize_fishers and fishers is not None:
        norm_constants = [_l2_norm_of_fisher(f) for f in fishers]
    else:
        norm_constants = None

    # The first model in the list of mergeable models is the "target" model and
    # the rest are "donor" models.
    output_model = clone_model(mergeable_models[0], cfg)
    output_variables = get_mergeable_variables(output_model)
    variables_to_merge = [get_mergeable_variables(m) for m in mergeable_models]

    # Make sure that all of the variable lists contain exactly the same number
    # of variables.
    assert len({len(output_variables)} | set(len(v) for v in variables_to_merge)) == 1

    for coefficients in coefficients_set:
        _merge_with_coeffs(
            output_variables,
            variables_to_merge,
            coefficients=coefficients,
            fishers=fishers,
            fisher_floor=fisher_floor,
            favor_target_model=favor_target_model,
            normalization_constants=norm_constants,
        )
        yield coefficients, output_model


def evaluate_model(merged_model, dataset, metric):
    # TODO
    return 0


def merging_coefficients_search(
    mergeable_models,
    coefficients_set,
    dataset,
    metric,
    fishers=None,
    fisher_floor: float = 1e-6,
    favor_target_model=True,
    normalize_fishers=True,
    print_results=True,
):
    merged_models = generate_merged_for_coeffs_set(
        mergeable_models,
        coefficients_set,
        fishers,
        fisher_floor=fisher_floor,
        favor_target_model=favor_target_model,
        normalize_fishers=normalize_fishers,
    )
    results = []

    for coeffs, merged_model in merged_models:
        score = evaluate_model(merged_model, dataset, metric)
        result = MergeResult(coefficients=coeffs, score=score)
        results.append(result)
        if print_results:
            print_merge_result(result)

    return results


def merging_models_fisher(
        cfg,
        mergeable_models,
        fishers=None,
        fisher_floor=1e-6,
        favor_target_model=True,
        normalize_fishers=True,
):
    coefficients_set = [[1.0 for i in range(len(mergeable_models))]]

    merged_models = generate_merged_for_coeffs_set(
        cfg,
        mergeable_models,
        coefficients_set,
        fishers,
        fisher_floor=fisher_floor,
        favor_target_model=favor_target_model,
        normalize_fishers=normalize_fishers,
    )

    _, merged_model = next(merged_models)

    return merged_model


def merging_models_isotropic(
        cfg,
        mergeable_models,
):
    # The first model in the list of mergeable models is the "target" model and
    # the rest are "donor" models.
    output_model = clone_model(mergeable_models[0], cfg)
    output_variables = get_mergeable_variables(output_model)
    variables_to_merge = [get_mergeable_variables(m) for m in mergeable_models]

    for i, var in enumerate(output_variables):
        s = 0
        for mvars in variables_to_merge:
            mvar = mvars[i]
            s += mvar.data

        # closed-form form solution of argmax
        var.data = s / len(mergeable_models)

    return output_model

