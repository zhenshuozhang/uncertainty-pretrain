import argparse

from utils.loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import gpytorch

from tqdm import tqdm
import numpy as np

from models.model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
from models.GP import GPModel, ExactGPModel
from utils.splitters import scaffold_split
from utils.scaler import StandardScaler
from utils.util import get_free_gpu
from base.train.DKL import *
from base.train.utils import cls_train, get_pred, write_result
from base.uncertainty.sgld import SGLDOptimizer
from base.uncertainty.focal_loss import SigmoidFocalLoss
from base.repsentation.rep_analysis import *

from copy import deepcopy
import pandas as pd

import os
import shutil



from tensorboardX import SummaryWriter

from torchmetrics.functional.classification import binary_calibration_error
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    auc,
    brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibrationDisplay

global_criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer, loss_fn=None):
    model.train()
    if loss_fn is not None:
        criterion = loss_fn
    else:
        criterion = global_criterion
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()

"""
def classification_metrics(preds, lbs):

    result_metrics_dict = dict()

    roc_auc_list = list()
    prc_auc_list = list()
    ece_list = list()
    mce_list = list()
    nll_list = list()
    brier_list = list()

    roc_auc_valid_flag = True
    prc_auc_valid_flag = True
    ece_valid_flag = True
    mce_valid_flag = True
    nll_valid_flag = True
    brier_valid_flag = True

    for i in range(lbs.shape[-1]):
        
        if torch.sum(lbs[:,i] == 1) > 0 and torch.sum(lbs[:,i] == -1) > 0:
            is_valid = lbs[:,i]**2 > 0
            lbs_ = (lbs[is_valid,i] + 1)/2
            preds_ = preds[is_valid,i]
        
            #lbs_ = lbs[:, i]
            #preds_ = preds[:, i]
            #lbs_ = (lbs_ + 1)/2
            if len(lbs_) < 1:
                continue
            if (lbs_ < 0).any():
                raise ValueError("Invalid label value encountered!")
            if (lbs_ == 0).all() or (lbs_ == 1).all():  # skip tasks with only one label type, as Uni-Mol did.
                continue
            preds_ = torch.sigmoid(preds_)
            # --- roc-auc ---
            try:
                roc_auc = roc_auc_score(lbs_, preds_)
                roc_auc_list.append(roc_auc)
            except Exception as e:
                roc_auc_valid_flag = False
                print("roc-auc error: ", e)

            # --- prc-auc ---
            try:
                p, r, _ = precision_recall_curve(lbs_, preds_)
                prc_auc = auc(r, p)
                prc_auc_list.append(prc_auc)
            except Exception as e:
                prc_auc_valid_flag = False
                print("prc-auc error: ", e)

            # --- ece ---
            try:
                ece = binary_calibration_error(preds_, lbs_).item()
                ece_list.append(ece)
                prob_true, prob_pred = calibration_curve(preds_, lbs_, n_bins=10)
                disp = CalibrationDisplay(prob_true, prob_pred, lbs_)
                disp.savefig('calibration.jpg')
            except Exception as e:
                ece_valid_flag = False
                print("ece error: ", e)

            # --- mce ---
            try:
                mce = binary_calibration_error(preds_, lbs_, norm='max').item()
                mce_list.append(mce)
            except Exception as e:
                mce_valid_flag = False
                print("mce error: ", e)

            # --- nll ---
            try:
                nll = F.binary_cross_entropy(
                    input=preds_,
                    target=lbs_.to(torch.float),
                    reduction='mean'
                ).item()
                nll_list.append(nll)
            except Exception as e:
                print("nll error: ", e)
                nll_valid_flag = False

            # --- brier ---
            try:
                brier = brier_score_loss(lbs_, preds_)
                brier_list.append(brier)
            except Exception as e:
                brier_valid_flag = False
                print("brier error: ", e)

    if roc_auc_valid_flag:
        roc_auc_avg = np.mean(roc_auc_list)
        result_metrics_dict['roc-auc'] = {'all': roc_auc_list, 'macro-avg': roc_auc_avg}

    if prc_auc_valid_flag:
        prc_auc_avg = np.mean(prc_auc_list)
        result_metrics_dict['prc-auc'] = {'all': prc_auc_list, 'macro-avg': prc_auc_avg}

    if ece_valid_flag:
        ece_avg = np.mean(ece_list)
        result_metrics_dict['ece'] = {'all': ece_list, 'macro-avg': ece_avg}

    if mce_valid_flag:
        mce_avg = np.mean(mce_list)
        result_metrics_dict['mce'] = {'all': mce_list, 'macro-avg': mce_avg}

    if nll_valid_flag:
        nll_avg = np.mean(nll_list)
        result_metrics_dict['nll'] = {'all': nll_list, 'macro-avg': nll_avg}

    if brier_valid_flag:
        brier_avg = np.mean(brier_list)
        result_metrics_dict['brier'] = {'brier': brier_list, 'macro-avg': brier_avg}

    return result_metrics_dict
"""

def cls_finetune(model, train_loader, val_loader, test_loader, ue_method, device, args):

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    if ue_method == 'none':
        for epoch in range(args.epochs):
            cls_train(args, model, device, train_loader, optimizer)
        show_rep_snr(model, test_loader, device, args, tag='nn')

        y_true, y_pred = get_pred(args, model, device, test_loader)
        result_metrics_dict = classification_metrics(y_pred, y_true)

    elif ue_method == 'focal':
        loss_fn = SigmoidFocalLoss(reduction='none')
        for epoch in range(args.epochs):
            cls_train(args, model, device, train_loader, optimizer, loss_fn)
        y_true, y_pred = get_pred(args, model, device, test_loader)
        result_metrics_dict = classification_metrics(y_pred, y_true)

    elif ue_method == 'sgld':
        for epoch in range(args.epochs):
            cls_train(args, model, device, train_loader, optimizer)
        sgld_optimizer = SGLDOptimizer(
            model.graph_pred_linear.parameters(), lr=args.lr, norm_sigma=0.1
        )
        n_sgld = 10
        sgld_y_pred = []
        for e in range(n_sgld):
            cls_train(args, model, device, train_loader, sgld_optimizer)
            y_true, y_pred = get_pred(args, model, device, test_loader)
            sgld_y_pred.append(np.array(y_pred))
        #print(sgld_y_pred)
        y_pred = torch.tensor(sgld_y_pred).mean(dim=0)
        result_metrics_dict = classification_metrics(y_pred, y_true)

    elif ue_method == 'ensemble':
        init_state_dict = model.state_dict()
        n_ensemble = args.n_ensemble

        ensemble_y_pred = []
        for ensemble in range(n_ensemble):
            model.load_state_dict(init_state_dict)
            model_param_group = []
            model_param_group.append({"params": model.gnn.parameters()})
            if args.graph_pooling == "attention":
                model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
            model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
            optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
            for e in range(args.epochs):
                cls_train(args, model, device, train_loader, optimizer)
            y_true, y_pred = get_pred(args, model, device, test_loader)

            ensemble_y_pred.append(np.array(y_pred))

        y_pred = torch.tensor(ensemble_y_pred).mean(dim=0)
        result_metrics_dict = classification_metrics(y_pred, y_true)

    elif ue_method == 'dkl':
        y_pred, y_var, y_true = dkl(model, train_loader, val_loader, test_loader, device, args)
        result_metrics_dict = classification_metrics(y_pred, y_true)

    elif ue_method == 'svdkl':
        y_pred, y_var, y_true = multi_single_svdkl(model, train_loader, val_loader, test_loader, device, args)
        result_metrics_dict = classification_metrics(y_pred, y_true, note=f'{args.dataset}_dkl')
    elif ue_method == 'embsvdkl':
        y_pred, y_var, y_true, _ = embsvdkl(model, train_loader, val_loader, test_loader, device, args)
        result_metrics_dict = classification_metrics(y_pred, y_true, note=f'{args.dataset}_embdkl')
    
    elif ue_method == 'fdkl':
        y_pred, y_var, y_true = fdkl(model, train_loader, val_loader, test_loader, device, args)
        result_metrics_dict = classification_metrics(y_pred, y_true)


    else:
        raise ValueError("Invalid uncertainty method")
    
    print(result_metrics_dict)
    #write_result(ue_method, result_metrics_dict, args)

    return result_metrics_dict
