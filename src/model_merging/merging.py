# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol Garcia Recasens
# CROMAI  ---  (pol.garcia@bsc.es)
# Barcelona Supercomputing Center (BSC)

import torch
from .model import get_mergeable_variables


def merging_models_fisher(
        output_model,
        mergeable_models,
        fishers=None,
        fisher_floor=1e-20,
        favor_target_model=True):
    output_variables = get_mergeable_variables(output_model
                                               )
    variables_to_merge = [get_mergeable_variables(m) for m in mergeable_models]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert len({len(output_variables)} | set(len(v) for v in variables_to_merge)) == 1

    n_models = len(variables_to_merge)

    if fishers is None:
        fishers = n_models * [1.0]
    else:
        assert len(fishers) == n_models

    d = dict(output_model.named_parameters())

    for idx, k in enumerate(list(d.keys())):
        # iterate over models
        s = torch.zeros_like(output_model.get_parameter(k)).to(device) 
        s_fisher = torch.zeros_like(output_model.get_parameter(k)).to(device)

        for m in range(len(mergeable_models)):
            diag = fishers[m].to(device) if isinstance(fishers[m], float) else fishers[m][idx].to(device)
            s = torch.add(s, mergeable_models[m].get_parameter(k)*diag)

            if not favor_target_model or m == 0:
                # ensure that fisher diagonal doesn't vanish
                diag = torch.clamp(diag, min=fisher_floor, max=float("inf"))
            s_fisher = torch.add(s_fisher, diag)
        
        d[k] = s / s_fisher

    output_model.load_state_dict(d, strict=False)
       
    return output_model.to(device)


def merging_models_isotropic(
        output_model,
        mergeable_models,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = dict(output_model.named_parameters())
    for idx, k in enumerate(list(d.keys())):
        s = torch.zeros_like(output_model.get_parameter(k)).to(device)
        for m in mergeable_models:
            s = torch.add(s, m.get_parameter(k))

        d[k].data = s / len(mergeable_models)
    
    output_model.load_state_dict(d, strict=False)

    return output_model.to(device)
