import argparse

from utils.loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import gpytorch
from scipy.stats import norm as gaussian

from tqdm import tqdm
import numpy as np

from models.model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from models.GP import GPModel, ExactGPModel
from utils.splitters import scaffold_split
from utils.scaler import StandardScaler
from utils.util import get_free_gpu
from base.train.DKL import dkl, svdkl, multi_single_svdkl, embsvdkl
from base.train.utils import reg_train, get_pred, write_result
from base.train.metrics import regression_metrics
from base.uncertainty.sgld import SGLDOptimizer
from base.uncertainty.focal_loss import SigmoidFocalLoss

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

global_criterion = nn.BCEWithLogitsLoss(reduction = "none")


def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        if args.dataset in ['qm7', 'qm8', 'qm9']:
            loss = torch.sum(torch.abs(pred-y))/y.size(0)
        elif args.dataset in ['esol','freesolv','lipophilicity']:
            loss = torch.sum((pred-y)**2)/y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def get_pred(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu()
    y_scores = torch.cat(y_scores, dim = 0).cpu()

    return y_true, y_scores


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()

    mse = mean_squared_error(y_true, y_scores)
    mae = mean_absolute_error(y_true, y_scores)
    rmse=np.sqrt(mean_squared_error(y_true,y_scores))
    return mse, mae, rmse


def reg_finetune(model, train_loader, val_loader, test_loader, ue_method, device, args):

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    result_metrics_dict = dict()

    if ue_method == 'none' or ue_method == 'ensemble':
        init_state_dict = model.state_dict()
        n_ensemble = args.n_ensemble

        ensemble_y_pred = []
        for ensemble in range(n_ensemble):
            model.load_state_dict(init_state_dict)
            for epoch in range(args.epochs):
                reg_train(args, model, device, train_loader, optimizer)
            y_true, y_pred = get_pred(args, model, device, test_loader)

            ensemble_y_pred.append(np.array(y_pred.numpy()))
        y_pred = torch.tensor(ensemble_y_pred).mean(dim=0)
        y_var = torch.tensor(ensemble_y_pred).var(dim=0)
        result_metrics_dict = regression_metrics(y_pred, y_var, y_true)

    elif ue_method == 'sgld':
        for epoch in range(args.epochs):
            reg_train(args, model, device, train_loader, optimizer)
        sgld_optimizer = SGLDOptimizer(
            model.graph_pred_linear.parameters(), lr=args.lr, norm_sigma=0.1
        )
        n_sgld = 10
        sgld_y_pred = []
        for e in range(n_sgld):
            reg_train(args, model, device, train_loader, sgld_optimizer)
            y_true, y_pred = get_pred(args, model, device, test_loader)
            sgld_y_pred.append(np.array(y_pred))
        #print(sgld_y_pred)
        y_pred = torch.tensor(sgld_y_pred).mean(dim=0)
        y_var = torch.tensor(sgld_y_pred).var(dim=0)
        result_metrics_dict = regression_metrics(y_pred, y_var, y_true)

    elif ue_method == 'dkl':
        y_pred, y_var, y_true = dkl(model, train_loader, val_loader, test_loader, device, args)
        result_metrics_dict = regression_metrics(y_pred, y_var, y_true)
    
    elif ue_method == 'svdkl':
        y_pred, y_var, y_true = multi_single_svdkl(model, train_loader, val_loader, test_loader, device, args)
        result_metrics_dict = regression_metrics(y_pred, y_var, y_true)
    elif ue_method == 'embsvdkl':
        y_pred, y_var, y_true, _, _ = embsvdkl(model, train_loader, val_loader, test_loader, device, args)
        result_metrics_dict = regression_metrics(y_pred, y_var, y_true)

    else:
        raise ValueError("Invalid uncertainty method")
    
    print(result_metrics_dict)
    #write_result(ue_method, result_metrics_dict, args)

    return result_metrics_dict
