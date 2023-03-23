# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Federico Bergamin, Pablo Moreno-Munoz
#                    Søren Hauberg, Georgios Arvanitidis
# CogSys Section  ---  {fedbe,pabmo,sohau,gear}@dtu.dk
# Technical University of Denmark (DTU)

import numpy as np
from sklearn import metrics
import torch
import math
from netcal.metrics import ECE, MCE

'''
File to store the metrics used in the experiments.
Credits to: https://github.com/runame/laplace-refinement/blob/308a3ff1f16b69dcb5bcea6ea302cf986e07350b/utils/metrics.py
'''

def accuracy(y_pred, y_true):
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        return np.mean(y_pred.argmax(1) == y_true).mean()*100


def nll(y_pred, y_true):
    """
    Mean Categorical negative log-likelihood. `y_pred` is a probability vector.
    """
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        return metrics.log_loss(y_true, y_pred)

def brier(y_pred, y_true):
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        def one_hot(targets, nb_classes):
            targets = targets.astype(int)
            res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
            return res.reshape(list(targets.shape)+[nb_classes])

        return metrics.mean_squared_error(y_pred, one_hot(y_true, y_pred.shape[-1]))
    

# def calibration(pys, y_true, M=100):
#     try:
#         pys, y_true = pys.detach().cpu().numpy(), y_true.cpu().numpy()
#     finally:
#         # Put the confidence into M bins
#         _, bins = np.histogram(pys, M, range=(0, 1))

#         labels = pys.argmax(1)
#         confs = np.max(pys, axis=1)
#         conf_idxs = np.digitize(confs, bins)

#         # Accuracy and avg. confidence per bin
#         accs_bin = []
#         confs_bin = []
#         nitems_bin = []

#         for i in range(M):
#             labels_i = labels[conf_idxs == i]
#             y_true_i = y_true[conf_idxs == i]
#             confs_i = confs[conf_idxs == i]

#             # print(confs_i)

#             acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
#             conf = np.nan_to_num(np.mean(confs_i), 0)
#             # acc = np.nanmean(labels_i == y_true_i)
#             # conf = np.nanmean(confs_i)  

#             accs_bin.append(acc)
#             confs_bin.append(conf)
#             nitems_bin.append(len(labels_i))

#         accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
#         nitems_bin = np.array(nitems_bin)

#         ECE = np.average(np.abs(confs_bin-accs_bin), weights=nitems_bin/nitems_bin.sum())
#         MCE = np.max(np.abs(accs_bin - confs_bin))

#         # In percent
#         ECE, MCE = ECE*100, MCE*100

#         return ECE, MCE

# Laplace Redux -- computation of ECE, MCE
def calibration(pys, y_true, M=15):
    
    # for binary classification it has to be 1-D prob array
    if y_true.max() == 1:
        pys = pys[:,1]

    _ECE = ECE(bins=M).measure(pys.cpu().numpy(), y_true.cpu().numpy())*100
    _MCE = MCE(bins=M).measure(pys.cpu().numpy(), y_true.cpu().numpy())*100
    return _ECE, _MCE
    

def nlpd_using_predictions(mu_star, var_star, true_target):
    nlpd = torch.abs(0.5*math.log(2*math.pi) + 0.5*torch.mean(torch.log(var_star) + (true_target - mu_star)**2/var_star))
    return nlpd

def mae(mu_star, true_target):
    mae = torch.mean(torch.abs(true_target - mu_star))
    return mae

def rmse(mu_star, true_target):
    rmse = torch.sqrt(torch.mean((true_target - mu_star)**2))
    return rmse


def error_metrics(mu_star, var_star, true_target):
    _rmse = rmse(mu_star, true_target)
    _mae = mae(mu_star, true_target)
    _nlpd = nlpd_using_predictions(mu_star, var_star, true_target)
    return _rmse, _mae, _nlpd


def compute_metrics(args, model, weights_list, test_data, verbose=True, save=None, device='cpu'):
    X_test = test_data['X'].to(device)
    y_test = test_data['y'].to(device)

    metrics_dict = {'accuracy': [], 
                    'nll': [],
                    'brier': [],
                    'ece': [],
                    'mce': []
                    }

    for weights in weights_list:
        p_y_test = 0
        for s in range(args.n_posterior_samples):
            # put the weights in the model
            torch.nn.utils.vector_to_parameters(weights[s,:].float(), model.parameters())
            # compute the predictions
            with torch.no_grad():
                p_y_test += torch.softmax(model(X_test), dim=1)

        p_y_test /= args.n_posterior_samples

        _accuracy = accuracy(p_y_test, y_test)
        _nll = nll(p_y_test, y_test)
        _brier = brier(p_y_test, y_test)
        _ece, _mce = calibration(p_y_test, y_test, M=args.calibration_bins)

        metrics_dict['accuracy'].append(_accuracy)
        metrics_dict['nll'].append(_nll)
        metrics_dict['brier'].append(_brier)
        metrics_dict['ece'].append(_ece)
        metrics_dict['mce'].append(_mce)

    if verbose:
        print_metrics(args, metrics_dict)

    if save is not None:
        np.save(save +'_metrics.npy', metrics_dict)
    else:
        print_metrics(args, metrics_dict)

    # return metrics_dict

def compute_metrics_per_sample(args, model, weights_list, test_data, verbose=True, device='cpu'):
    X_test = test_data['X'].to(device)
    y_test = test_data['y'].to(device)

    metrics_list = []
    for weights in weights_list:
        p_y_test = 0
        metrics_dict = {'accuracy': [], 
                        'nll': [],
                        'brier': [],
                        'ece': [],
                        'mce': []
                        }

        for s in range(args.n_posterior_samples):
            # put the weights in the model
            torch.nn.utils.vector_to_parameters(weights[s,:].float(), model.parameters())
            # compute the predictions
            with torch.no_grad():
                p_y_test = torch.softmax(model(X_test), dim=1)

            _accuracy = accuracy(p_y_test, y_test)
            _nll = nll(p_y_test, y_test)
            _brier = brier(p_y_test, y_test)
            _ece, _mce = calibration(p_y_test, y_test, M=args.calibration_bins)

            metrics_dict['accuracy'].append(_accuracy)
            metrics_dict['nll'].append(_nll)
            metrics_dict['brier'].append(_brier)
            metrics_dict['ece'].append(_ece)
            metrics_dict['mce'].append(_mce)
        
        metrics_list.append(metrics_dict)

    if verbose:
        print_metrics_per_sample(args, metrics_list)
        
    return metrics_list

def print_metrics(args, metrics_dict):
    for metric, m_list in metrics_dict.items():
        print('>', metric, m_list)

def print_metrics_per_sample(args, metrics_list):
    for metrics_dict in metrics_list:
        for metric, m_list in metrics_dict.items():
            print('>', metric, m_list)
