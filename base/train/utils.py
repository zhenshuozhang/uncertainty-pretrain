import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import gpytorch
from torch.utils.tensorboard import SummaryWriter

from sklearn.cluster import KMeans

from base.train.metrics import classification_metrics, task_cls_metrics, task_reg_metrics

import shutil

def reg_train(args, model, device, loader, optimizer, loss_fn=None):
    model.train()

    if loss_fn is None:
        if args.dataset in ['qm7', 'qm8', 'qm9']:
            loss_fn = lambda x, y: torch.sum(torch.abs(x-y))/y.size(0)
        elif args.dataset in ['esol','freesolv','lipophilicity']:
            loss_fn = lambda x, y: torch.sum((x-y)**2)/y.size(0)

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def cls_train(args, model, device, loader, optimizer, loss_fn=None):
    model.train()
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss(reduction = "none")
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        #pred = torch.sigmoid(pred)
        loss_mat = loss_fn(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()

def get_pred(args, model, device, loader):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu()
    y_pred = torch.cat(y_pred, dim = 0).cpu()

    return y_true, y_pred

def write_result(ue_method, result_metrics_dict, args):
    if args.task_type == 'cls':
        file = 'cls_result.log'
    elif args.task_type == 'reg':
        file = 'reg_result.log'
    with open(file, 'a+') as f:
        f.write(f' dataset: {args.dataset}\n uncertainty method: {ue_method}\n')
        f.write(str(args))
        f.write('\n')
        for key, value in result_metrics_dict.items():
            f.write(key + ' ' + str(value))
            f.write('\n')
        f.write('\n')

def cls_train_stat(model, likelihood, ls, task_index, loader, epoch, device, args, tag='test'):

    model.eval()
    likelihood.eval()
    path = f"tsb/{args.dataset}/{tag}"
    writer_embed = SummaryWriter(path)

    writer_embed.add_scalar('length scale', ls, epoch)

    y_true = []
    y_pred = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)
        
        with torch.no_grad():
            pred = model.nn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_pred.append(pred)
    
    y_true = torch.cat(y_true, dim = 0).cpu()
    y_pred = torch.cat(y_pred, dim = 0).cpu()
    result_metrics_dict_gnn = task_cls_metrics(y_pred, y_true, task_index)
    gnn_acc = result_metrics_dict_gnn['roc-auc']['macro-avg']
    gnn_u = result_metrics_dict_gnn['ece']['macro-avg']
    writer_embed.add_scalar('gnn acc', gnn_acc, epoch)
    writer_embed.add_scalar('gnn u', gnn_u, epoch)
    
    
    # test DKL
    y_pred = []
    y_var = []
    y_true = []
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)

            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            y=batch.y.view(-1, args.num_tasks)[:, task_index].view(-1, 1)
            y_true.append(y)
            y_pred.append(pred.mean)
            y_var.append(torch.diagonal(pred.covariance_matrix))

    y_true = torch.cat(y_true, dim = 0).cpu()
    y_pred = torch.cat(y_pred, dim=0)
    y_var = torch.cat(y_var, dim=0)
    result_metrics_dict_dkl = task_cls_metrics(y_pred.unsqueeze(1).cpu(), y_true.cpu(), task_index)
    dkl_acc = result_metrics_dict_dkl['roc-auc']['macro-avg']
    dkl_u = result_metrics_dict_dkl['ece']['macro-avg']
    writer_embed.add_scalar('dkl acc', dkl_acc, epoch)
    writer_embed.add_scalar('dkl u', dkl_u, epoch)
    
    model.train()
    likelihood.train()

def reg_train_stat(model, likelihood, ls, task_index, loader, epoch, device, args, tag='test'):

    model.eval()
    likelihood.eval()
    path = f"tsb/{args.dataset}/{tag}"
    writer_embed = SummaryWriter(path)

    writer_embed.add_scalar('length scale', ls, epoch)
    
    y_true = []
    y_pred = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)
        
        with torch.no_grad():
            pred = model.nn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_pred.append(pred)
    
    y_true = torch.cat(y_true, dim = 0).cpu()
    y_pred = torch.cat(y_pred, dim = 0).cpu()
    result_metrics_dict_gnn = task_reg_metrics(y_pred, None, y_true, task_index)
    gnn_acc = result_metrics_dict_gnn['rmse']['macro-avg']
    #gnn_u = result_metrics_dict_gnn['ce']['macro-avg']
    writer_embed.add_scalar('gnn acc', gnn_acc, epoch)
    #writer_embed.add_scalar('gnn u', gnn_u, epoch)
    
    
    # test DKL
    y_pred = []
    y_var = []
    y_true = []
    
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32):
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)

            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            y=batch.y.view(-1, args.num_tasks)[:, task_index].view(-1, 1)
            y_true.append(y)
            y_pred.append(pred.mean)
            y_var.append(torch.diagonal(pred.covariance_matrix))

    y_true = torch.cat(y_true, dim = 0).cpu()
    y_pred = torch.cat(y_pred, dim=0)
    y_var = torch.cat(y_var, dim=0)
    result_metrics_dict_dkl = task_reg_metrics(y_pred.unsqueeze(1).cpu(), y_var.cpu(), y_true.cpu(), task_index)
    dkl_acc = result_metrics_dict_dkl['rmse']['macro-avg']
    dkl_u = result_metrics_dict_dkl['ce']['macro-avg']
    writer_embed.add_scalar('dkl acc', dkl_acc, epoch)
    writer_embed.add_scalar('dkl u', dkl_u, epoch)
    
    model.train()
    likelihood.train()


def is_model_equal(model1, model2, device):
    equal = True
    model1 = model1.to(device)
    model2 = model2.to(device)

    p1 = []
    for p in model1.parameters():
        p1.append(p)
    p2 = []
    for p in model2.parameters():
        p2.append(p)
    
    if len(p1) == len(p2):
        for i in range(len(p1)):
            if not torch.equal(p1[i], p2[i]):
                equal = False
                break
    else:
        equal = False

    print(equal)
    return equal

def calc_inducing_points(model, train_loader, task_index, type, n_inducing_points, device, per_cluster_lengthscale=True):
    print("inducing point init: ", type)
    print("inducing point num: ", n_inducing_points)
    embs = []
    preds = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            output = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embs.append(output.cpu())
            preds.append(pred.cpu())
        embs = torch.cat(embs).cpu()
        preds = torch.cat(preds).cpu()

        if type == "mult-gp":
            per_atom_emb = {}
            max_atoms = 0
            for batch in tqdm(train_loader):
                embedding = model(batch.cuda())
                for i in range(max(batch.batch) + 1):
                    current = embedding[batch.batch == i]
                    max_atoms = max(max_atoms, len(current))
                    for j in range(max_atoms):
                        if j in per_atom_emb:
                            per_atom_emb[j].append(current[j:j+1, :])
                        else:
                            per_atom_emb[j] = [current[j:j+1, :]]
                            
            inducing_points = []
            lengthscales = []
            num_per_cluster = n_inducing_points // max_atoms
            for k in per_atom_emb:
                embs = torch.cat(per_atom_emb[k])
                mask = np.random.choice(len(embs), size=num_per_cluster)
                inducing_points.append(embs[mask])
                if per_cluster_lengthscale:
                    try: 
                        lengthscales.append(torch.pdist(embs.cpu()).mean())
                    except:
                        lengthscales.append(torch.pdist(embs.cpu()[torch.randint(0, len(embs), (10000,))]).mean())
                        
            inducing_points = torch.cat(inducing_points)
            
            if per_cluster_lengthscale:
                # calculate lengthscale only within each cluster
                initial_lengthscale = torch.mean(torch.tensor(lengthscales))
            else:
                # calculate lengthscales also between the clusters
                initial_lengthscale = torch.pdist(inducing_points)
        elif type == "k-means" or type == "kmeans":
            data = embs.numpy()
            # use embeddings to get inducing points
            kmeans = KMeans(n_clusters=n_inducing_points).fit(data)
            inducing_points = torch.tensor(kmeans.cluster_centers_)
            inducing_points = inducing_points.to(device)
            inducing_points = model.lin_pred(inducing_points)[:, task_index].view(-1, 1)
            inducing_points.cpu()
            # use pred result to get lengthscale
            preds = preds[:, task_index].view(-1, 1)
            try:
                #initial_lengthscale = torch.pdist(embs.cpu()).mean()
                initial_lengthscale = torch.pdist(preds.cpu()).mean()
            except Exception as e:
                print('k-means error: ', e)
                initial_lengthscale = torch.pdist(embs.cpu()[torch.randint(0, len(embs), (50000,))]).mean()
            
            
        elif type == "first":
            inducing_points = torch.cat(embs)[:n_inducing_points]
            initial_lengthscale = torch.pdist(inducing_points.cpu()).mean()
        elif type == "random":
            mask = np.random.choice(embs.shape[0], size=n_inducing_points)
            inducing_points = preds[mask, task_index].view(-1, 1)
            initial_lengthscale = torch.pdist(inducing_points.cpu()).mean()
        else:
            raise NotImplementedError
        #initial_lengthscale /= 10
    return inducing_points, initial_lengthscale

def calc_emb_inducing_points(model, train_loader, task_index, type, n_inducing_points, device, per_cluster_lengthscale=True):
    print("inducing point init: ", type)
    print("inducing point num: ", n_inducing_points)
    embs = []
    preds = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            output = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embs.append(output.cpu())
            preds.append(pred.cpu())
        embs = torch.cat(embs).cpu()
        preds = torch.cat(preds).cpu()

        if type == "mult-gp":
            per_atom_emb = {}
            max_atoms = 0
            for batch in tqdm(train_loader):
                embedding = model(batch.cuda())
                for i in range(max(batch.batch) + 1):
                    current = embedding[batch.batch == i]
                    max_atoms = max(max_atoms, len(current))
                    for j in range(max_atoms):
                        if j in per_atom_emb:
                            per_atom_emb[j].append(current[j:j+1, :])
                        else:
                            per_atom_emb[j] = [current[j:j+1, :]]
                            
            inducing_points = []
            lengthscales = []
            num_per_cluster = n_inducing_points // max_atoms
            for k in per_atom_emb:
                embs = torch.cat(per_atom_emb[k])
                mask = np.random.choice(len(embs), size=num_per_cluster)
                inducing_points.append(embs[mask])
                if per_cluster_lengthscale:
                    try: 
                        lengthscales.append(torch.pdist(embs.cpu()).mean())
                    except:
                        lengthscales.append(torch.pdist(embs.cpu()[torch.randint(0, len(embs), (10000,))]).mean())
                        
            inducing_points = torch.cat(inducing_points)
            
            if per_cluster_lengthscale:
                # calculate lengthscale only within each cluster
                initial_lengthscale = torch.mean(torch.tensor(lengthscales))
            else:
                # calculate lengthscales also between the clusters
                initial_lengthscale = torch.pdist(inducing_points)
        elif type == "k-means" or type == "kmeans":
            data = embs.numpy()
            # use embeddings to get inducing points
            kmeans = KMeans(n_clusters=n_inducing_points).fit(data)
            inducing_points = torch.tensor(kmeans.cluster_centers_)
            inducing_points = inducing_points.to(device)
            #inducing_points = model.lin_pred(inducing_points)[:, task_index].view(-1, 1)
            inducing_points.cpu()
            # use pred result to get lengthscale
            preds = preds[:, task_index].view(-1, 1)
            try:
                initial_lengthscale = torch.pdist(embs.cpu()).mean()
                #initial_lengthscale = torch.pdist(preds.cpu()).mean()
            except Exception as e:
                print('k-means error: ', e)
                initial_lengthscale = torch.pdist(embs.cpu()[torch.randint(0, len(embs), (50000,))]).mean()
            
            
        elif type == "first":
            inducing_points = torch.cat(embs)[:n_inducing_points]
            initial_lengthscale = torch.pdist(inducing_points.cpu()).mean()
        elif type == "random":
            mask = np.random.choice(embs.shape[0], size=n_inducing_points)
            inducing_points = embs[mask]
            initial_lengthscale = torch.pdist(inducing_points.cpu()).mean()
        else:
            raise NotImplementedError
        #initial_lengthscale *= 2
    return inducing_points, initial_lengthscale
