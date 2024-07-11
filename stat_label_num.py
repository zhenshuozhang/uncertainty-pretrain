import argparse

from utils.loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from models.model import GNN, GNN_graphpred
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

from utils.util import get_free_gpu, load_best_configs

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

def stat_label_num(loader, num_tasks):
    num_total = []
    for i in range(num_tasks):
        num_lb = [0, 0]
        y_true = []
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            y = batch.y.view(-1, num_tasks)
            y = ((y+1)/2)[:, i]
            num_lb[0] += (y==0).sum().item()
            num_lb[1] += (y==1).sum().item()
            y_true.append(y)

        #y_true = torch.cat(y_true, dim = 0)
        print(num_lb)
        num_total.append(num_lb)
    return num_total


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=512,
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

    parser.add_argument('--csv', action='store_true', help='write result to csv file')
    parser.add_argument('--use_cfg', action='store_true', help='use configs')
    parser.add_argument('--tsb', action='store_true', help='tensorboard')

    parser.add_argument('--save', action='store_true', help='save finetuned model')
    parser.add_argument('--load', action='store_true', help='load finetuned model')

    parser.add_argument('--tag', type=str, default = 'none', help='save tag')
    args = parser.parse_args()

    if args.use_cfg:
        #cfg_path = f'./configs/{args.dataset}.yml'
        args = load_best_configs(args, './configs/config.yml')
    print(args)
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        free_gpu = get_free_gpu()
        device = 'cuda:{}'.format(free_gpu)
    else:
        device = 'cpu'
    
    result_metrics_dict_all = []
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    dataset = MoleculeDataset("/data/zzs/chem-dataset/dataset/" + args.dataset, dataset=args.dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('/data/zzs/chem-dataset/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('/data/zzs/chem-dataset/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print('train size: ', len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    stat_label_num(test_loader, num_tasks)

    
if __name__ == "__main__":
    main()
