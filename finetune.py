import argparse

from utils.loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from models.model import GNN, GNN_graphpred, GNN_proj_graphpred
from models.GP import *
from sklearn.metrics import roc_auc_score

from utils.splitters import scaffold_split
from utils.scaler import StandardScaler
from utils.util import write_result, save_results_csv
from base.uncertainty.sgld import SGLDOptimizer
from base.train.cls_finetune import cls_finetune
from base.train.reg_finetune import reg_finetune
from base.repsentation.rep_analysis import *
import pandas as pd

import os
import shutil

from utils.util import get_free_gpu, load_best_configs, stat_label_num

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

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    model.train()

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

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    return y_true, y_scores

def pred_eval(pred, y_true):
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

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            lbs_ = (y_true[is_valid,i] + 1)/2
            preds_ = pred[is_valid,i]

            roc_list.append(roc_auc_score(lbs_, pred[is_valid,i]))
            preds_ = torch.sigmoid(torch.from_numpy(preds_))
            ece = binary_calibration_error(preds_, torch.from_numpy(lbs_)).item()
            ece_list.append(ece)
            brier = brier_score_loss(lbs_, preds_)
            brier_list.append(brier)
            mce = binary_calibration_error(preds_, torch.from_numpy(lbs_), norm='max').item()
            mce_list.append(mce)
            nll = F.binary_cross_entropy(
                input=preds_,
                target=torch.from_numpy(lbs_).to(torch.float),
                reduction='mean'
            ).item()
            nll_list.append(nll)
    
    if ece_valid_flag:
        ece_avg = np.mean(ece_list)
        result_metrics_dict['ece'] = {'all': ece_list, 'macro-avg': ece_avg}

    if mce_valid_flag:
        mce_avg = np.mean(mce_list)
        result_metrics_dict['mce'] = {'all': mce_list, 'macro-avg': mce_avg}
    
    if brier_valid_flag:
        brier_avg = np.mean(brier_list)
        result_metrics_dict['brier'] = {'brier': brier_list, 'macro-avg': brier_avg}
    
    if nll_valid_flag:
        nll_avg = np.mean(nll_list)
        result_metrics_dict['nll'] = {'all': nll_list, 'macro-avg': nll_avg}
    
    #print(result_metrics_dict)

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list), result_metrics_dict #y_true.shape[1]

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

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    #print(y_scores)
    #print(y_true)

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

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            lbs_ = (y_true[is_valid,i] + 1)/2
            preds_ = y_scores[is_valid,i]

            roc_list.append(roc_auc_score(lbs_, y_scores[is_valid,i]))
            preds_ = torch.sigmoid(torch.from_numpy(preds_))
            ece = binary_calibration_error(preds_, torch.from_numpy(lbs_)).item()
            ece_list.append(ece)
            brier = brier_score_loss(lbs_, preds_)
            brier_list.append(brier)
            mce = binary_calibration_error(preds_, torch.from_numpy(lbs_), norm='max').item()
            mce_list.append(mce)
            nll = F.binary_cross_entropy(
                input=preds_,
                target=torch.from_numpy(lbs_).to(torch.float),
                reduction='mean'
            ).item()
            nll_list.append(nll)
    
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
    
    #print(result_metrics_dict)

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list), result_metrics_dict #y_true.shape[1]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--decay_dkl', type=float, default=0,
                        help='dkl weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--proj_dim', type=int, default=30,
                        help='projection dimensions (default: 30)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'bbbp', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'models_graphcl/graphcl_80.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=-1, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--runs', type=int, default=1, help = "runs")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 1, help='number of workers for dataset loading')
    parser.add_argument('--fix', type=bool, default = False, help='number of workers for dataset loading')

    parser.add_argument('--ue_method', type=str, default = 'none', help='uncertainty method')
    parser.add_argument('--n_ensemble', type=int, default = 10, help='number of workers for dataset loading')

    parser.add_argument('--dkl_epochs', type=int, default = 50, help='dkl epochs')
    parser.add_argument('--dkl_s', type=str, default ='s2', help='dkl parameters strategy')
    parser.add_argument('--n_inducing_points', type=int, default = 64, help='number of inducing points')
    parser.add_argument('--ind_type', type=str, default ='kmeans', help='inducing points strategy')
    parser.add_argument('--dkl_scale', type=float, default = 1, help='dkl parameters strategy')
    parser.add_argument('--grid_bounds', type=float, default = 10, help='dkl parameters strategy')
    parser.add_argument('--grid_size', type=int, default = 32, help='dkl parameters strategy')
    parser.add_argument('--ll', type=str, default ='elbo', help='likelihood function')
    parser.add_argument('--extra_pll', action='store_true', help='load finetuned model')

    parser.add_argument('--csv', action='store_true', help='write result to csv file')
    parser.add_argument('--use_cfg', action='store_true', help='use configs')
    parser.add_argument('--use_exp', action='store_true', help='use the exp configs')
    parser.add_argument('--tsb', action='store_true', help='tensorboard')

    parser.add_argument('--save', action='store_true', help='save finetuned model')
    parser.add_argument('--load', action='store_true', help='load finetuned model')
    parser.add_argument('--save_kernel', action='store_true', default=True, help='save kernel')

    parser.add_argument('--tag', type=str, default = 'none', help='save tag')
    args = parser.parse_args()

    if args.use_cfg:
        #cfg_path = f'./configs/{args.dataset}.yml'
        args = load_best_configs(args, './configs/config.yml')
    if args.use_exp:
        args = load_best_configs(args, './configs/config_exp.yml')
    print(args)

    if torch.cuda.is_available():
        free_gpu = get_free_gpu()
        device = 'cuda:{}'.format(free_gpu)
    else:
        device = 'cpu'
    #device='cpu'
    args.device = device
    
    result_metrics_dict_all = []
    seeds = range(args.runs)
    #seeds = [1,2,4,0]
    for run in range(args.runs):
        if args.seed != -1 and args.runs > 0:
            seed = args.seed
        else:
            seed = seeds[run]
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        """
        if args.dataset in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag']:
            task_type = 'cls'
        else:
            task_type = 'reg'
        
        args.task_type = task_type

        #Bunch of classification tasks
        if args.dataset == "tox21":
            num_tasks = 12
        elif args.dataset == "pcba":
            num_tasks = 128
        elif args.dataset == "bace":
            num_tasks = 1
        elif args.dataset == "bbbp":
            num_tasks = 1
        elif args.dataset == "toxcast":
            num_tasks = 617
        elif args.dataset == "sider":
            num_tasks = 27
        elif args.dataset == "clintox":
            num_tasks = 2
        elif args.dataset == "hiv":
            num_tasks = 1
        elif args.dataset == "muv":
            num_tasks = 17
        elif args.dataset == 'esol':
            num_tasks = 1
        elif args.dataset == 'freesolv':
            num_tasks = 1
        elif args.dataset == 'lipophilicity':
            num_tasks = 1
        elif args.dataset == 'qm7':
            num_tasks = 1
        elif args.dataset == 'qm8':
            num_tasks = 12
        elif args.dataset == 'qm9':
            num_tasks = 12
        else:
            raise ValueError("Invalid dataset name.")
        args.num_tasks = num_tasks
        #set up dataset
        dataset = MoleculeDataset("/home1/zzs/chem-dataset/dataset/" + args.dataset, dataset=args.dataset)
    
        if args.split == "scaffold":
            smiles_list = pd.read_csv('/home1/zzs/chem-dataset/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
            print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
            print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv('/home1/zzs/chem-dataset/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
            print("random scaffold")
        else:
            raise ValueError("Invalid split option.")
        
        if args.tsb:
            path = f"tsb/{args.dataset}"
            try:
                shutil.rmtree(path)
            except:
                pass

        print('train size: ', len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        #set up model
        model = GNN_proj_graphpred(args.num_layer, args.emb_dim, args.proj_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        #model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        if not args.input_model_file == "none":
            backbone = 'GraphCL'
            model.from_pretrained(args.input_model_file)
        else:
            backbone = 'nopretrain'
        print(device)
        model.to(device)
        if task_type == 'cls':
            show_rep_snr(model, test_loader, device, args, tag='init')
        if task_type == 'cls':
            result_metrics_dict = cls_finetune(model, train_loader, val_loader, test_loader, args.ue_method, device, args)
        else:
            result_metrics_dict = reg_finetune(model, train_loader, val_loader, test_loader, args.ue_method, device, args)
        
        result_metrics_dict_all.append(result_metrics_dict)
    
    results = dict()
    # register metrics key
    for key, value in result_metrics_dict_all[0].items():
        results[key] = {'all': [], 'avg': 0}
    # record
    for result_item in result_metrics_dict_all:
        for key, value in result_item.items():
            results[key]['all'].append(value['macro-avg'])
    # calculate avg
    for key, value in result_metrics_dict_all[0].items():
        results[key]['avg'] = np.mean(results[key]['all'])
    
    print(results)
    write_result(args.ue_method, results, args)
    if args.tag != 'none':
        note = args.dkl_s + '-' + args.tag
    else:
        note = args.dkl_s
    if args.csv:
        save_results_csv(results, args.ue_method, backbone, args.dataset, task_type, note=note)

    
if __name__ == "__main__":
    main()
