import argparse

from utils.loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset

from tqdm import tqdm
import numpy as np

from models.model import GNN, GNN_graphpred, GNN_proj_graphpred
from models.GP import *
from models.svdkl import SVDKL, embSVDKL
from sklearn.metrics import roc_auc_score

from utils.splitters import scaffold_split
from utils.loader import MoleculeDataset_aug
from utils.scaler import StandardScaler
from base.train.utils import cls_train, reg_train, get_pred, cls_train_stat, is_model_equal, reg_train_stat
from base.uncertainty.sgld import SGLDOptimizer
from base.train.cls_finetune import cls_finetune
from base.train.reg_finetune import reg_finetune
from base.train.metrics import regression_metrics
from base.repsentation.rep_analysis import *
import pandas as pd

import os
import os.path as op
import shutil

from utils.util import get_free_gpu, load_best_configs, stat_label_num

def save_results_csv(results, args, note='none'):
    result_dir = f'./saved_models/{args.dataset}/'
    metrics = list(results.keys())
    title = f'{args.dis_type} distance'
    columns_headers = [title] + [f"{metric}" for metric in metrics]
    columns = {k: list() for k in columns_headers}
    if note != 'none':
        columns[title] = f"{args.input_model_file}_{note}"
    else:
        columns[title] = f"{args.input_model_file}"

    for key, value in results.items():
        if value is None:
            value = 'no-value'
        if isinstance(value, str):
            columns[key] = [value]
        else:
            columns[key] = ["{:.4f}".format(value)]

    df = pd.DataFrame(columns)
    csv_path = f'{result_dir}/result.csv'
    if op.exists(csv_path):
        mode = 'a'
        header = False
    else:
        mode = 'w'
        header = True
    df.to_csv(csv_path, mode=mode, header=header, index=False)
    return None

def cal_dis(data1, data2, kernel, type='kernel'):
    if type == 'kernel':
        with torch.no_grad():
            d = kernel(data1, data2).to_dense()
    elif type == 'l2':
        #d = torch.dist(data1, data2, p=2)
        #d = torch.sqrt(torch.sum((data1-data2)**2, dim=1))
        d = torch.sqrt(torch.mm((data1-data2), (data1-data2).T))
    elif type == 'cos':
        norm_data1 = F.normalize(data1)
        norm_data2 = F.normalize(data2)
        d = 1 - torch.mm(norm_data1, norm_data2.T)
    return d

def cal_deep_kernel(data1, data2, model):
    kernel = model.gp.covar_module
    p1 = model.proj_from_enc(data1)
    p2 = model.proj_from_enc(data2)
    d = kernel(p1, p2).to_dense()

    return d

"""
use loss to find center point & farthest edge point
return center embedding & edge embedding
"""
def find_center_edge(model, data_loader, task, args):
    model.eval()
    device = model.device

    loss_func = None
    # accuracy loss
    if loss_func is None:
        if args.dataset in ['qm7', 'qm8', 'qm9']:
            loss_fn = lambda x, y: torch.abs(x-y)
        elif args.dataset in ['esol','freesolv','lipophilicity']:
            loss_fn = lambda x, y: (x-y)**2
    # ELBO likelihood loss
    elif loss_func == 'ELBO':
        loss_fn = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp, num_data=len(data_loader.dataset))
    # PLL loss
    elif loss_func == 'pll':
        loss_fn = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model.gp, num_data=len(data_loader.dataset))

    # scan the dataloader & find the corresponding embedding of min and max loss 
    min_loss = 1e7
    max_loss = -1e7

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch = batch.to(device)

            with torch.no_grad():
                output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, task)
                emb = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model.likelihood(output)
            y=batch.y.to(torch.float64).view(-1, args.num_tasks)[:, task]

            if loss_func is None:
                loss = loss_fn(pred.mean, y)
            else:
                loss = -loss_fn(output, y)

            batch_min_loss, batch_min_index = loss.min(), loss.argmin()
            batch_max_loss, batch_max_index = loss.max(), loss.argmax()
            
            if batch_min_loss < min_loss:
                center_point = emb[batch_min_index]
            if batch_max_loss > max_loss:
                edge_point = emb[batch_max_index]
    
    return center_point.unsqueeze(0), edge_point.unsqueeze(0)

"""
have kernel of 1st data only
calculate the distance of each point in 2nd data and the center & edge data in 1st data
"""
def uni_kernel_single_all_stat(model, data_loader1, data_loader2, task, args):
    model.eval()
    device = model.device
    #dis_type = 'l2'
    #dis_type = 'kernel'
    dis_type = args.dis_type

    center_point, edge_point = find_center_edge(model, data_loader1, task, args)
    kernel = model.gp.covar_module

    in_dis = 0
    out_dis = 0

    self_emb_list = []
    self_dis_list = []
    self_dis_edge_list = []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for step, batch in enumerate(tqdm(data_loader1, desc="Iteration")):
            batch = batch.to(device)

            with torch.no_grad():
                emb = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                #pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, task)
            #output = model.likelihood(pred)
            
            # delete center point
            dis = cal_dis(emb, center_point, kernel, type=dis_type)
            dis_edge = cal_dis(emb, edge_point, kernel, type=dis_type)
            self_emb_list.append(emb)
            self_dis_list.append(dis)
            self_dis_edge_list.append(dis_edge)
    self_emb_all = torch.cat(self_emb_list, dim=0)
    self_dis_all = torch.cat(self_dis_list, dim=0)
    self_dis_edge_all = torch.cat(self_dis_edge_list, dim=0)

    radius = self_dis_all.min()
    edge_dis = cal_dis(edge_point, center_point, kernel, type=dis_type)

    emb_list = []
    dis_list = []
    dis_edge_list = []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for step, batch in enumerate(tqdm(data_loader2, desc="Iteration")):
            batch = batch.to(device)

            with torch.no_grad():
                emb = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, task)
            output = model.likelihood(pred)
            dis = cal_dis(emb, center_point, kernel, type=dis_type)
            dis_edge = cal_dis(emb, edge_point, kernel, type=dis_type)

            """
            for i in range(dis.shape[0]):
                if dis[i] > radius:
                    in_dis += 1
                else:
                    out_dis +=1
            """

            if dis_type == 'kernel':
                r = self_dis_all.min()
                in_dis += (dis>r).sum().item()
                out_dis += (dis<r).sum().item()
            elif dis_type == 'l2':
                r = self_dis_all.max()
                in_dis += (dis<r).sum().item()
                out_dis += (dis>r).sum().item()

            emb_list.append(emb)
            dis_list.append(dis)
            dis_edge_list.append(dis_edge)
    emb_all = torch.cat(emb_list, dim=0)
    dis_all = torch.cat(dis_list, dim=0)
    dis_edge_all = torch.cat(dis_edge_list, dim=0)

    file = f'./saved_models/uni_kernel_single_all_result_{args.dataset}.log'
    
    with open(file, 'a+') as f:
        f.write(f'dataset: {args.dataset}\n distance type: {dis_type}\n model: {args.m}\n')
        f.write(f' kernel output scale: {kernel.outputscale.detach()}\n')
        #f.write(str(args))
        #f.write('\n')
        f.write(f'  self points:\n   min: {self_dis_all.min()}, max: {self_dis_all.max()}, mean: {self_dis_all.mean()}, edge: {edge_dis.item()}\n')
        f.write(f'  center dis:\n   min: {dis_all.min()}, max: {dis_all.max()}, mean: {dis_all.mean()}\n')
        f.write(f'  in dis: {in_dis}, out dis: {out_dis}\n')
        f.write('\n')
    
    print(kernel.outputscale)
    print('distance type: ', dis_type)
    print('radius: ', radius)
    print("dis all")
    print(dis_all.mean())
    print(dis_all.min())
    print(dis_all.max())
    print("dis egde")
    print(dis_edge_all.mean())
    print(dis_edge_all.min())
    print(dis_edge_all.max())
    print('stat')
    print(in_dis)
    print(out_dis)

"""
have kernel of 1st data only
calculate the distribution distance of 1st data and 2nd data
using sampled MMD distance
"""
def uni_kernel_all_all_stat(model, data_loader1, data_loader2, task, args):
    model.eval()
    device = model.device
    dis_type = args.dis_type
    print(dis_type)
    print(device)
    #center_point, edge_point = find_center_edge(model, data_loader1, task, args)
    kernel = model.gp.covar_module

    emb1_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader1, desc="Iteration")):
            batch = batch.to(device)
            with torch.no_grad():
                emb = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
                #emb = model.enc_wo_proj(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
            emb1_list.append(emb)
        emb1 = torch.cat(emb1_list, dim=0)

        m = emb1.shape[0]
        #k1 = kernel(emb1, emb1).to_dense()
        k1 = cal_dis(emb1, emb1, kernel, type=dis_type)
        #term1 = (k1.sum() - torch.trace(k1))/(2*m*(m-1))
        term1 = torch.triu(k1).sum()/(m*(m+1)/2)
        print(term1)

        emb2_list = []
        for step, batch in enumerate(tqdm(data_loader2, desc="Iteration")):
            batch = batch.to(device)
            with torch.no_grad():
                emb = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            emb2_list.append(emb)
        emb2 = torch.cat(emb2_list, dim=0)

        
        total_n = emb2.shape[0]
        sample_size = m
        sample_num = int(total_n / m) + 1
        sample_list = []
        t1_list = []
        t2_list = []
        t3_list = []

        
        for s in tqdm(range(sample_num)):
            mask = torch.randperm(total_n)[:sample_size]
            sample_emb2 = deepcopy(emb2[mask])
            n = sample_emb2.shape[0]
            #k2 = kernel(sample_emb2, sample_emb2).to_dense()
            k2 = cal_dis(sample_emb2, sample_emb2, kernel, type=dis_type)
            #term2 = (k2.sum() - torch.trace(k2))/(2*n*(n-1))
            term2 = torch.triu(k2).sum()/(n*(n+1)/2)
            #k3 = kernel(emb1, sample_emb2).to_dense()
            k3 = cal_dis(emb1, sample_emb2, kernel, type=dis_type)
            #term3 = -(k3.sum() + torch.trace(k3))/(m*n)
            term3 = -2*(k3.sum()/(m*n))
            mmd2_sample = term1 + term2 + term3
            t = (mmd2_sample).to('cpu').numpy()
            sample_list.append(t)
            t2_list.append(term2.to('cpu').numpy())
            t3_list.append(term3.to('cpu').numpy())

        print('sample num: ', len(sample_list))
        mmd2_mean = np.mean(sample_list)
        t2_mean = np.mean(t2_list)
        t3_mean = np.mean(t3_list)
        print(mmd2_mean)

        return term1.to('cpu').numpy(), t2_mean, t3_mean, mmd2_mean

"""
have kernel of 1st data only
calculate the distribution distance of 1st data and 2nd data
using sampled MMD distance
"""
def uni_kernel_all_batch_stat(model, data_loader1, batch2, task, args):
    model.eval()
    device = model.device
    dis_type = args.dis_type
    print(dis_type)
    print(device)
    #center_point, edge_point = find_center_edge(model, data_loader1, task, args)
    kernel = model.gp.covar_module

    emb1_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader1, desc="Iteration")):
            batch = batch.to(device)
            with torch.no_grad():
                emb = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
                #emb = model.enc_wo_proj(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
            emb1_list.append(emb)
        emb1 = torch.cat(emb1_list, dim=0)

        m = emb1.shape[0]
        #k1 = kernel(emb1, emb1).to_dense()
        k1 = cal_dis(emb1, emb1, kernel, type=dis_type)
        #term1 = (k1.sum() - torch.trace(k1))/(2*m*(m-1))
        term1 = torch.triu(k1).sum()/(m*(m+1)/2)
        print(term1)

        batch2 = batch2.to(device)
        with torch.no_grad():
            emb2 = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        n = emb2.shape[0]
        #k2 = kernel(sample_emb2, sample_emb2).to_dense()
        k2 = cal_dis(emb2, emb2, kernel, type=dis_type)
        #term2 = (k2.sum() - torch.trace(k2))/(2*n*(n-1))
        term2 = torch.triu(k2).sum()/(n*(n+1)/2)
        #k3 = kernel(emb1, sample_emb2).to_dense()
        k3 = cal_dis(emb1, emb2, kernel, type=dis_type)
        #term3 = -(k3.sum() + torch.trace(k3))/(m*n)
        term3 = -2*(k3.sum()/(m*n))
        mmd2 = term1 + term2 + term3

        return term1.to('cpu').numpy(), term2.to('cpu').numpy(), term3.to('cpu').numpy(), mmd2.to('cpu').numpy()

"""
single kernel, same data, different encoder
"""
def uni_deep_kernel_uni_stat(encoder1, encoder2, model, data_loader, task, args, sample=False):
    model.eval()
    device = model.device
    dis_type = args.dis_type
    print(dis_type)
    print(device)
    #center_point, edge_point = find_center_edge(model, data_loader1, task, args)

    if not sample:
        with torch.no_grad():
            emb1_list = []
            emb2_list = []
            for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
                batch = batch.to(device)
                with torch.no_grad():
                    emb1 = encoder1.enc_wo_proj(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
                    emb2 = encoder2.enc_wo_proj(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
                emb1_list.append(emb1)
                emb2_list.append(emb2)
            emb1 = torch.cat(emb1_list, dim=0)
            emb2 = torch.cat(emb2_list, dim=0)

            m = emb1.shape[0]
            #k1 = kernel(emb1, emb1).to_dense()
            k1 = cal_deep_kernel(emb1, emb1, model)
            #term1 = (k1.sum() - torch.trace(k1))/(2*m*(m-1))
            term1 = torch.triu(k1).sum()/(m*(m+1)/2)

            n = emb2.shape[0]
            k2 = cal_deep_kernel(emb2, emb2, model)
            term2 = torch.triu(k2).sum()/(n*(n+1)/2)

            k3 = cal_deep_kernel(emb1, emb2, model)
            term3 = -2*(k3.sum()/(m*n))

            mmd2 = term1 + term2 + term3
            term1 = term1.to('cpu').numpy()
            term2 = term2.to('cpu').numpy()
            term3 = term3.to('cpu').numpy()
            mmd2 = mmd2.to('cpu').numpy()
    else:
        term1_list = []
        term2_list = []
        term3_list = []
        mmd2_list = []
        with torch.no_grad():
            emb1_list = []
            emb2_list = []
            for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
                batch = batch.to(device)
                with torch.no_grad():
                    emb1 = encoder1.enc_wo_proj(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
                    emb2 = encoder2.enc_wo_proj(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
                m = emb1.shape[0]
                #k1 = kernel(emb1, emb1).to_dense()
                k1 = cal_deep_kernel(emb1, emb1, model)
                #term1 = (k1.sum() - torch.trace(k1))/(2*m*(m-1))
                term1 = torch.triu(k1).sum()/(m*(m+1)/2)

                n = emb2.shape[0]
                k2 = cal_deep_kernel(emb2, emb2, model)
                term2 = torch.triu(k2).sum()/(n*(n+1)/2)

                k3 = cal_deep_kernel(emb1, emb2, model)
                term3 = -2*(k3.sum()/(m*n))

                mmd2 = term1 + term2 + term3

                term1_list.append(term1.to('cpu').numpy())
                term2_list.append(term2.to('cpu').numpy())
                term3_list.append(term3.to('cpu').numpy())
                mmd2_list.append(mmd2.to('cpu').numpy())
        
        term1 = np.mean(term1_list)
        term2 = np.mean(term2_list)
        term3 = np.mean(term3_list)
        mmd2 = np.mean(mmd2_list)

    return term1, term2, term3, mmd2

"""
have kernel of 1st data only
calculate the distribution distance of 1st data and 2nd data
using sampled MMD distance
"""
def uni_kernel_all_all_sample_stat(model, data_loader1, data_loader2, task, args):
    model.eval()
    device = model.device
    dis_type = args.dis_type
    print(dis_type)

    #center_point, edge_point = find_center_edge(model, data_loader1, task, args)
    kernel = model.gp.covar_module

    emb1_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader1, desc="Iteration")):
            batch = batch.to(device)
            with torch.no_grad():
                emb = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
            emb1_list.append(emb)
        emb1 = torch.cat(emb1_list, dim=0)

        emb2_list = []
        for step, batch in enumerate(tqdm(data_loader2, desc="Iteration")):
            batch = batch.to(device)
            with torch.no_grad():
                emb = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            emb2_list.append(emb)
        emb2 = torch.cat(emb2_list, dim=0)

        
        total_n = emb2.shape[0]
        total_m = emb1.shape[0]
        sample_size = min(10000, total_m)
        print(sample_size)
        sample_num = int(total_n / sample_size) + 1
        sample_list = []

        
        for s in tqdm(range(sample_num)):
            mask1 = torch.randperm(total_m)[:sample_size]
            mask2 = torch.randperm(total_n)[:sample_size]

            sample_emb1 = deepcopy(emb1[mask1])
            m = sample_emb1.shape[0]
            k1 = cal_dis(sample_emb1, sample_emb1, kernel, type=dis_type)
            term1 = torch.triu(k1).sum()/(m*m)

            sample_emb2 = deepcopy(emb2[mask2])
            n = sample_emb2.shape[0]
            #k2 = kernel(sample_emb2, sample_emb2).to_dense()
            k2 = cal_dis(sample_emb2, sample_emb2, kernel, type=dis_type)
            #term2 = (k2.sum() - torch.trace(k2))/(2*n*(n-1))
            term2 = torch.triu(k2).sum()/(n*n)

            #k3 = kernel(emb1, sample_emb2).to_dense()
            k3 = cal_dis(sample_emb1, sample_emb2, kernel, type=dis_type)
            #term3 = -(k3.sum() + torch.trace(k3))/(m*n)
            term3 = -2*torch.triu(k3).sum()/(m*n)
            mmd2_sample = term1 + term2 + term3
            t = (mmd2_sample).to('cpu').numpy()
            sample_list.append(t)

        print('sample num: ', len(sample_list))
        mmd2_mean = np.mean(sample_list)
        print(mmd2_mean)

        return mmd2_mean

def multi_kernel_single_data_stat(model1, model2, data_loader, task1, task2, args):
    model1.eval()
    model2.eval()
    device = model1.device

    center_point_1, edge_point_1 = find_center_edge(model1, data_loader, task1, args)
    center_point_2, edge_point_2 = find_center_edge(model2, data_loader, task2, args)
    kernel_1 = model1.gp.covar_module
    kernel_2 = model2.gp.covar_module



def MMD(kernel1, kernel2):
    k1 = torch.mm(kernel1, kernel1)
    k2 = torch.mm(kernel2, kernel2)
    k1k2 = torch.mm(kernel1, kernel2)
    loss = torch.trace(k1) + torch.trace(k2) - 2*torch.trace(k1k2)

    return loss

def MDS(D, n=2):
    def cal_B(D):
        (n1, n2) = D.shape
        DD = np.square(D)                    # 矩阵D 所有元素平方
        Di = np.sum(DD, axis=1) / n1         # 计算dist(i.)^2
        Dj = np.sum(DD, axis=0) / n1         # 计算dist(.j)^2
        Dij = np.sum(DD) / (n1 ** 2)         # 计算dist(ij)^2
        B = np.zeros((n1, n1))
        for i in range(n1):
            for j in range(n2):
                B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)   # 计算b(ij)
        return B

    D = D.to('cpu').numpy()
    B = cal_B(D)
    Be, Bv = np.linalg.eigh(B)             # Be矩阵B的特征值，Bv归一化的特征向量
    # print np.sum(B-np.dot(np.dot(Bv,np.diag(Be)),Bv.T))
    Be_sort = np.argsort(-Be)
    Be = Be[Be_sort]                          # 特征值从大到小排序
    Bv = Bv[:, Be_sort]                       # 归一化特征向量
    Bez = np.diag(Be[0:n])                 # 前n个特征值对角矩阵
    # print Bez
    Bvz = Bv[:, 0:n]                          # 前n个归一化特征向量
    Z = np.dot(np.sqrt(Bez), Bvz.T).T
    Z = torch.from_numpy(Z)
    return Z

def total_process_1(data_loader1, data_loader2, args):
    device = args.device
    model = embSVDKL(args.num_layer, args.emb_dim, args.proj_dim, args.num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, device=args.device, n_inducing_points=args.n_inducing_points, dkl_s=args.dkl_s)
    
    # process kernel metric
    # pretrain
    args.dis_type = 'kernel'
    args.m = 'pre'
    model_file = f'./saved_models/{args.dataset}/model_{args.task}.pth'
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model = model.to(device)
    uni_kernel_single_all_stat(model, data_loader1, data_loader2, task=0, args=args)
    # no-pretrain
    args.m = 'none'
    model_file = f'./saved_models/{args.dataset}/model_{args.task}_none.pth'
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model = model.to(device)
    uni_kernel_single_all_stat(model, data_loader1, data_loader2, task=0, args=args)

    # process l2 metric
    # pretrain
    args.dis_type = 'l2'
    args.m = 'pre'
    model_file = f'./saved_models/{args.dataset}/model_{args.task}.pth'
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model = model.to(device)
    uni_kernel_single_all_stat(model, data_loader1, data_loader2, task=0, args=args)
    # no-pretrain
    args.m = 'none'
    model_file = f'./saved_models/{args.dataset}/model_{args.task}_none.pth'
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model = model.to(device)
    uni_kernel_single_all_stat(model, data_loader1, data_loader2, task=0, args=args)

def total_process_2(data_loader, test_loader, in_data_loader, out_data_loader, all_data_loader, result, args):
    device = args.device
    model = embSVDKL(args.num_layer, args.emb_dim, args.proj_dim, args.num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, device=args.device, n_inducing_points=args.n_inducing_points, dkl_s=args.dkl_s)
    task = 0
    # process kernel metric
    # pretrain
    if args.input_model_file != 'none':
        args.m = 'pre'
        parts = args.input_model_file.rsplit('/', 1)
        model_name = parts[-1]
        model_file = f'./saved_models/{args.dataset}/model_{task}_{model_name}'
    else:
        model_file = f'./saved_models/{args.dataset}/model_{task}_none.pth'
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model = model.to(device)

    ctr_test = 'pred'
    if ctr_test == 'dkl':
        # test DKL
        y_pred_i = []
        y_var_i = []
        y_true_i = []
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
                batch = batch.to(device)

                with torch.no_grad():
                    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, task)
                output = model.likelihood(pred)
                y=batch.y.view(-1, args.num_tasks)[:, task].view(-1, 1)
                y_true_i.append(y)
                y_pred_i.append(pred.mean)
                y_var_i.append(torch.diagonal(pred.covariance_matrix).unsqueeze(1))

        y_true = torch.cat(y_true_i, dim=0).cpu()
        y_pred = torch.cat(y_pred_i, dim=0).cpu()
        y_var = torch.cat(y_var_i, dim=0).cpu()
    else:
        y_true, y_pred = get_pred(args, model.nn, device, test_loader)
        y_var = None

    result_metrics_dict = regression_metrics(y_pred, y_var, y_true)

    result['rmse'] = result_metrics_dict['rmse']['macro-avg']
    #result['nll'] = result_metrics_dict['nll']['macro-avg']
    result['nll'] = 0
    print(result)

    if args.input_model_file == 'none' or in_data_loader is None or out_data_loader is None:
        result['in_out'] = None
        result['train_in'] = None
        result['train_out'] = None
        result['train_all'] = uni_kernel_all_all_stat(model, data_loader, all_data_loader, task=0, args=args)
    else:
        result['in_out'] = uni_kernel_all_all_sample_stat(model, in_data_loader, out_data_loader, task=0, args=args)
        result['train_in'] = uni_kernel_all_all_stat(model, data_loader, in_data_loader, task=0, args=args)
        result['train_out'] = uni_kernel_all_all_stat(model, data_loader, out_data_loader, task=0, args=args)
        result['train_all'] = uni_kernel_all_all_stat(model, data_loader, all_data_loader, task=0, args=args)

    print(result)



def main():
    parser = argparse.ArgumentParser(description='analysis kernels')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--pretrain_batch_size', type=int, default=1024,
                        help='input batch size for pre-training (default: 32)')
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
    parser.add_argument('--proj_dim', type=int, default=300,
                        help='projection dimensions (default: 30)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default = 'models_graphcl/graphcl_80.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=-1, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--runs', type=int, default=1, help = "runs")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 1, help='number of workers for dataset loading')
    parser.add_argument('--fix', type=bool, default = False, help='number of workers for dataset loading')
    parser.add_argument('--pretrain_dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset', type=str, default = 'qm7', help='dataset')

    parser.add_argument('--task', type=int, default = 0, help='task')

    parser.add_argument('--dkl_epochs', type=int, default = 50, help='dkl epochs')
    parser.add_argument('--dkl_s', type=str, default ='s2', help='dkl parameters strategy')
    parser.add_argument('--n_inducing_points', type=int, default = 64, help='number of inducing points')
    parser.add_argument('--ind_type', type=str, default ='kmeans', help='inducing points strategy')
    parser.add_argument('--dkl_scale', type=float, default = 1, help='dkl parameters strategy')
    parser.add_argument('--grid_bounds', type=float, default = 10, help='dkl parameters strategy')
    parser.add_argument('--grid_size', type=int, default = 32, help='dkl parameters strategy')
    parser.add_argument('--ll', type=str, default ='elbo', help='likelihood function')
    parser.add_argument('--extra_pll', action='store_true', help='load finetuned model')

    parser.add_argument('--m', type=str, default ='pre', help='model')
    parser.add_argument('--dis_type', type=str, default ='kernel', help='distance metric')

    parser.add_argument('--sub_ratio', type=float, default = 1)

    parser.add_argument('--use_cfg', action='store_true', help='use configs')
    parser.add_argument('--use_exp', action='store_true', help='use the exp configs')

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
    dataset = MoleculeDataset("/docker-data/zzs/chem-dataset/dataset/" + args.dataset, dataset=args.dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('/docker-data/zzs/chem-dataset/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('/docker-data/zzs/chem-dataset/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print('train size: ', len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    pretrain_dataset = MoleculeDataset_aug("/docker-data/zzs/chem-dataset/dataset/" + args.pretrain_dataset, dataset=args.pretrain_dataset, aug='none')
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    #set up model
    model = embSVDKL(args.num_layer, args.emb_dim, args.proj_dim, args.num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, device=args.device, n_inducing_points=args.n_inducing_points, dkl_s=args.dkl_s)
    
    #total_process_1(test_loader, pretrain_loader, args)

    result = dict()
    result['input_model'] = args.input_model_file
    result['distance function'] = args.dis_type
    result['model_id'] = None
    result['size_of_in'] = len(pretrain_dataset)
    result['size_of_out'] = 0
    if args.input_model_file != 'none':
        if args.input_model_file.find('sub') != -1:
            parts = args.input_model_file.rsplit('graphcl_sub_', 1)
            model_id = parts[-1][:-4]
            result['model_id'] = model_id
            print(args.input_model_file)
            print(model_id)
            pretrain_indices = torch.load(f'./models_graphcl/indices_{model_id}.pt')
            mask = torch.Tensor([False for i in range(len(pretrain_dataset))])
            indices = torch.Tensor(range(len(pretrain_dataset))).int()
            mask[pretrain_indices] = True
            out_indices = indices[mask==0].tolist()
            result['size_of_in'] = len(pretrain_indices)
            result['size_of_out'] = len(out_indices)
            print('size of in: ', len(pretrain_indices))
            print('size of out: ', len(out_indices))
            in_sub_dataset = Subset(pretrain_dataset, pretrain_indices)
            out_sub_dataset = Subset(pretrain_dataset, out_indices)
            in_sub_loader = DataLoader(in_sub_dataset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
            out_sub_loader = DataLoader(out_sub_dataset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
            
            total_process_2(train_loader, test_loader, in_sub_loader, out_sub_loader, pretrain_loader, result, args)
        else:
            total_process_2(train_loader, test_loader, None, None, pretrain_loader, result, args)
    else:
        total_process_2(train_loader, test_loader, None, None, pretrain_loader, result, args)
    save_results_csv(result, args)
    #total_process_2(train_loader, test_loader, val_loader, args)
    """
    if args.m == 'none':
        model_file = f'./saved_models/{args.dataset}/model_{args.task}_none.pth'
    elif args.m == 'bad':
        model_file = f'./saved_models/{args.dataset}/model_{args.task}_bad.pth'
    else:
        model_file = f'./saved_models/{args.dataset}/model_{args.task}.pth'
    model.load_state_dict(torch.load(model_file, map_location='cpu'))

    model = model.to(device)

    print(f'size of pretrain data: {len(pretrain_dataset)}')
    print(f'size of finetune data: {len(train_dataset)}')

    uni_kernel_single_all_stat(model, train_loader, test_loader, task=0, args=args)
    #uni_kernel_single_all_stat(model, train_loader, pretrain_loader, task=0, args=args)

    #uni_kernel_all_all_stat(model, train_loader, test_loader, task=0, args=args)
    #uni_kernel_all_all_stat(model, train_loader, pretrain_loader, task=0, args=args)

    #mds = MDS(kernel1)
    #print(mds)
    """

if __name__ == "__main__":
    main()

