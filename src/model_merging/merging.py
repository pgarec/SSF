import collections
import torch
import torch.nn as nn
import torch.distributions as dist

from .model import clone_model, get_mergeable_variables, MLP


def merging_models_fisher(
        output_model,
        mergeable_models,
        fishers=None,
        fisher_floor=1e-20,
        favor_target_model=True):
    output_variables = get_mergeable_variables(output_model)
    variables_to_merge = [get_mergeable_variables(m) for m in mergeable_models]

    assert len({len(output_variables)} | set(len(v) for v in variables_to_merge)) == 1

    n_models = len(variables_to_merge)

    if fishers is None:
        fishers = n_models * [1.0]
    else:
        assert len(fishers) == n_models

    d = dict(output_model.named_parameters())

    for idx, k in enumerate(list(d.keys())):
        # iterate over models
        s = torch.zeros_like(output_model.get_parameter(k)) 
        s_fisher = torch.zeros_like(output_model.get_parameter(k))

        for m in range(len(mergeable_models)):
            diag = fishers[m] if isinstance(fishers[m], float) else fishers[m][idx]
            s = torch.add(s, mergeable_models[m].get_parameter(k)*diag)

            if not favor_target_model or m == 0:
                # ensure that fisher diagonal doesn't vanish
                diag = torch.clamp(diag, min=fisher_floor, max=float("inf"))
            s_fisher = torch.add(s_fisher, diag)
        
        d[k] = s / s_fisher

    output_model.load_state_dict(d, strict=False)
       
    return output_model



def merging_models_fisher_subsets(
        output_model,
        mergeable_models,
        fishers=None,
        n_classes=0,
        fisher_floor=1e-6,
        favor_target_model=True
        ):
    output_variables = get_mergeable_variables(output_model)
    variables_to_merge = [get_mergeable_variables(m) for m in mergeable_models]

    # Make sure that all of the variable lists contain exactly the same number
    # of variables.
    assert len({len(output_variables)} | set(len(v) for v in variables_to_merge)) == 1

    n_models = len(variables_to_merge)

    if fishers is None:
        fishers = n_models * [1.0]
        fishers = torch.ones((n_models, len(variables_to_merge[0])))
    else:
        assert len(fishers) == n_models

    d = dict(output_model.named_parameters())
    clf = []
    clf_key = []

    for idx, k in enumerate(list(d.keys())):
            s = torch.zeros_like(output_model.get_parameter(k)) 
            s_fisher = torch.zeros_like(output_model.get_parameter(k)) 

            for m in range(len(mergeable_models)):
                if 'clf' in k:
                    clf.append(mergeable_models[m].get_parameter(k))
                    clf_key = k
                else:
                    diag = fishers[m] if isinstance(fishers[m], float) else fishers[m][idx]
                    s = torch.add(s, mergeable_models[m].get_parameter(k)*diag)
                    if not favor_target_model or m == 0:
                        diag = torch.clamp(diag, min=fisher_floor, max=float("inf"))
                    s_fisher = torch.add(s_fisher, diag)
            
            if not 'clf' in k:
                d[k] = s / s_fisher
    
    # assumption that the models are ordered in the config file
    d[clf_key] = torch.cat(clf, dim=0)
    output_model.load_state_dict(d)
    # cfg.data.n_classes = int(cfg.data.n_classes/cfg.data.n_models)
    # print(cfg.data.n_classes)


    return output_model


def merging_models_isotropic(
        output_model,
        mergeable_models,
):
    d = dict(output_model.named_parameters())
    for idx, k in enumerate(list(d.keys())):
        s = torch.zeros_like(output_model.get_parameter(k)) 
        for m in mergeable_models:
            s = torch.add(s, m.get_parameter(k))

        d[k].data = s / len(mergeable_models)

    output_model.load_state_dict(d, strict=False)

    return output_model
