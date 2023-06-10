# Model-driven models

Open-source probabilistic framework for model merging written in PyTorch. Implemented during the master thesis at Cognitive Systems @ DTU. 

Permutation merging is a probabilistic merging approach that leverages permutations of parameters and conditional independence to achieve a more accurate construction of the posterior distribution. We empirically demonstrate that our merging technique outperforms the current state-of-the-art method, Fisher merging. By guiding the merging optimization with the joint likelihood of the models’ posterior distribution we are able to find an optimum parameter set without re-visiting any data.
