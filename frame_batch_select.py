import argparse
import os
from datetime import datetime

from utils.loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
#from torch.utils.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import random_split

from tqdm import tqdm
import numpy as np
import random

from models.model import GNN
from models.model import GNN_graphpred, GNN_proj_graphpred
from sklearn.metrics import roc_auc_score

from utils.splitters import scaffold_split, random_split, random_scaffold_split
from utils.util import get_free_gpu, load_best_configs, get_num_tasks
from utils.loader import MoleculeDataset
from base.train.cls_finetune import cls_finetune
from base.train.reg_finetune import reg_finetune
from base.train.DKL import dkl, svdkl, multi_single_svdkl, embsvdkl
from ana_kernel import uni_kernel_all_all_stat, uni_deep_kernel_uni_stat, uni_kernel_all_batch_stat
from base.train.metrics import regression_metrics, classification_metrics
import pandas as pd

from tensorboardX import SummaryWriter

from copy import deepcopy

class SelectedDataset(MoleculeDataset):
    def __init__(self, original_dataset, selected_indices):
        self.original_dataset = original_dataset
        self.selected_indices = selected_indices

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        original_idx = self.selected_indices[idx]
        return self.original_dataset[original_idx]

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def train(args, model, device, dataset, optimizer):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.pre_batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.pre_batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step+1)

def train_subset(args, model, device, dataset1, optimizer):

    dataset1.aug = "none"
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.pre_batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.pre_batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--pre_batch_size', type=int, default=2048,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--pre_epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--epochs', type=int, default=20,
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
    parser.add_argument('--proj_dim', type=int, default=300,
                        help='projection dimensions (default: 30)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--pre_dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset', type=str, default = 'qm7', help='downstream dataset')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'random')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default = 'random')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)

    parser.add_argument('--sub_ratio', type=float, default = 0.1)

    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')

    parser.add_argument('--dis_type', type=str, default ='kernel', help='distance metric')

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

    parser.add_argument('--order', type=str, default ='order', help='order or reverse')


    parser.add_argument('--save', action='store_true', help='save finetuned model')
    parser.add_argument('--load', action='store_true', help='load finetuned model')
    parser.add_argument('--save_kernel', action='store_true', default=False, help='save kernel')

    parser.add_argument('--tag', type=str, default = 'none', help='save tag')

    args = parser.parse_args()

    if args.use_cfg:
        args = load_best_configs(args, './configs/config_pre.yml')

    if args.dataset in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag']:
        task_type = 'cls'
    else:
        task_type = 'reg'
    args.task_type = task_type
    
    num_tasks = get_num_tasks(args.dataset)
    args.num_tasks = num_tasks

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

    #set up dataset
    print(args.pre_dataset)
    pre_dataset = MoleculeDataset_aug("/home1/zzs/chem-dataset/dataset/" + args.pre_dataset, dataset=args.pre_dataset)
    print(pre_dataset)

    print('train size: ', len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    pretrain_loader = DataLoader(pre_dataset, batch_size=args.pre_batch_size, num_workers = args.num_workers, shuffle=False)

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        free_gpu = get_free_gpu()
        #free_gpu = 3
        print(free_gpu)
        device = 'cuda:{}'.format(free_gpu)
    else:
        device = 'cpu'
    args.device = device
    #device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = graphcl(gnn)
    model.to(device)
    temp_model = graphcl(gnn).to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    

    iter_num = 10
    iter_epochs = int(args.pre_epochs / iter_num)
    #iter_epochs = 1
    total_samples = len(pre_dataset) 
    batch_size = pretrain_loader.batch_size
    num_batch = total_samples // batch_size + (total_samples % batch_size != 0)

    print(num_batch)

    select_num = int(num_batch * args.sub_ratio)

    file = f'./select_batch_stat/result_train_{args.dataset}_{iter_num}_{select_num}in{num_batch}.log'
    result = dict()
    try:
        for i in range(iter_num):
            
            # train dkl
            temp_model = GNN_proj_graphpred(args.num_layer, args.emb_dim, args.proj_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type).to(device)
            temp_model.gnn.load_state_dict(gnn.state_dict())
            if task_type == 'cls':

                y_pred, y_var, y_true, GNN_DKL, nn_acc = embsvdkl(temp_model, train_loader, val_loader, test_loader, device, args)
                temp_model.eval()
                result_metrics_dict = classification_metrics(y_pred, y_true, note=f'{args.dataset}_embdkl')    
                result['nn_acc'] = nn_acc
                result['acc'] = result_metrics_dict['roc-auc']['macro-avg']

            else:

                y_pred, y_var, y_true, GNN_DKL, nn_acc = embsvdkl(temp_model, train_loader, val_loader, test_loader, device, args)
                temp_model.eval()
                result_metrics_dict = regression_metrics(y_pred, y_var, y_true)
                result['nn_acc'] = nn_acc
                result['acc'] = result_metrics_dict['rmse']['macro-avg']
            
            select_index = []
            # subset selection: stat
            if args.order == 'random':
                subset_order = list(range(len(subsets_index)))
                random.shuffle(subset_order)
                random_subsets_index = [subsets_index[x] for x in subset_order]
                select_index = sum(random_subsets_index[:select_num], [])
            else:
                batch_mmd = []
                for batch_idx, batch in enumerate(pretrain_loader):
                    t1, t2, t3, mean_mmd2 = uni_kernel_all_batch_stat(GNN_DKL, train_loader, batch, 0, args)
                    batch_mmd.append(mean_mmd2)
                
                if args.order == 'order':
                    batch_order = np.argsort(batch_mmd)
                elif args.order == 'reverse':
                    batch_order = np.argsort(batch_mmd)[::-1]
                #batch_order = np.argsort(batch_mmd)
                
                """
                if args.order == 'reverse':
                    select_batch_index = batch_order[-select_num:]
                elif args.order == 'order':
                    select_batch_index = batch_order[:select_num]
                """
                select_batch_index = batch_order[:select_num]
                for bi in select_batch_index:
                    upper_bound = min((bi + 1) * pretrain_loader.batch_size, total_samples-1)
                    select_index.extend(range(bi * pretrain_loader.batch_size, upper_bound))

                #select_dataset = SelectedDataset(pre_dataset, select_index)
                random.shuffle(select_index)
                select_dataset = Subset(pre_dataset, select_index)

            now = datetime.now()
            with open(file, 'a+') as f:
                f.write(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Args: {args}\n")
                f.write(f'dataset: {args.dataset}\n distance type: {args.dis_type} order: {args.order}\n')
                f.write(f' iter: {i}\n')
                f.write(f'  finetune result: dkl result: {result["acc"]}, nn result: {result["nn_acc"]}\n')
                f.write(f'  subset batch index: dkl result: {select_batch_index}\n')
                f.write('\n')

            # train
            for e in range(iter_epochs):
                print(f"====iter {i} ====epoch " + str(e))
                #train_acc, train_loss = train(args, model, device, select_dataset, optimizer)
                train_acc, train_loss = train_subset(args, model, device, select_dataset, optimizer)
    except Exception as e:
        error_message = str(e)
        # 打开一个文件并写入异常信息
        now = datetime.now()
        with open(file, 'a+') as f:
            f.write(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(error_message)
            f.write('\n')
        
    torch.save(gnn.state_dict(), f"./models_graphcl/graphcl_batch_select_{args.dataset}_{iter_num}_{select_num}_order.pth")


if __name__ == "__main__":
    main()
