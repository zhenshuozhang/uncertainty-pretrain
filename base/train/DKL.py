import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import gpytorch
from sklearn.metrics import roc_auc_score

from models.GP import *
from models.svdkl import SVDKL, embSVDKL
from base.train.utils import cls_train, reg_train, get_pred, cls_train_stat, is_model_equal, reg_train_stat
from base.train.metrics import classification_metrics, regression_metrics
from base.repsentation.rep_analysis import *

from tqdm import tqdm

def save_kernel(pred, dataset, task, args):
    if not os.path.exists(f'./kernel/{dataset}'):
        os.mkdir(f'./kernel/{dataset}')
    if args.input_model_file == 'none':
        kernel_name = f'kernel_{task}_none'
    else:
        kernel_name = f'kernel_{task}'
    torch.save(pred, f'./kernel/{dataset}/{kernel_name}.pt')

def save_model(model, dataset, task, args):
    if not os.path.exists(f'./saved_models/{dataset}'):
        os.mkdir(f'./saved_models/{dataset}')
    parts = args.input_model_file.rsplit('/', 1)
    model_name = parts[-1]
    if args.input_model_file == 'none':
        model_file = f'./saved_models/{dataset}/model_{task}_none.pth'
    else:
        model_file = f'./saved_models/{dataset}/model_{task}_{model_name}'
    torch.save(model.to('cpu').state_dict(), model_file)

def get_parameters(model, gp_model, lr, dkl_s, args):
    gp_param_group = []
    if dkl_s=='s1':
        gp_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            gp_param_group.append({"params": model.pool.parameters(), "lr":lr})
        gp_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":lr})
    elif dkl_s=='s2':
        gp_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":lr})
    elif dkl_s=='s3':
        model.reset_pred()
        gp_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            gp_param_group.append({"params": model.pool.parameters(), "lr":lr})
        gp_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":lr})
    elif dkl_s=='s4':
        gp_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            gp_param_group.append({"params": model.pool.parameters(), "lr":lr})

    gp_param_group.append({"params": gp_model.parameters(), "lr":0.01})

    return gp_param_group

def train_nn(args, model, device, train_loader, test_loader):
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    model_file = f'./models_finetune/{args.dataset}.pth'
    if args.load:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
    else:
        for epoch in range(args.epochs):
            if args.task_type == 'cls':
                cls_train(args, model, device, train_loader, optimizer)
            elif args.task_type == 'reg':
                reg_train(args, model, device, train_loader, optimizer)

    if args.save:
        torch.save(model.state_dict(), model_file)

def get_nn_pred(args, model, device, loader):
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        #with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu()
    y_pred = torch.cat(y_pred, dim = 0).cpu()

    return y_true, y_pred

def dkl_process(gp_model, gp_optimizer, likelihood, mll, train_x, train_y, test_x, test_y, dkl_epochs, args):
    gp_model.train()
    likelihood.train()
    dkl_epochs = args.dkl_epochs
    for de in range(dkl_epochs):
        # Zero gradients from previous iteration
        gp_optimizer.zero_grad()
        # Output from model
        output = gp_model(train_x)
        # Calc loss and backprop gradients
        print(train_y.shape)
        loss = -mll(output, train_y).mean()
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f ' % (
            de + 1, dkl_epochs, loss.item(),
            gp_model.covar_module.base_kernel.lengthscale.item()
        ))
        gp_optimizer.step()
    gp_model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gp_model = gp_model.to('cpu')
        test_x = test_x.to('cpu')
        observed_pred = likelihood(gp_model(test_x))
    
    return observed_pred

def fdkl(model, train_loader, val_loader, test_loader, device, args):
    train_nn(args, model, device, train_loader)
    y_true, y_pred = get_pred(args, model, device, test_loader)

    result_metrics_dict = classification_metrics(y_pred, y_true)
    print(result_metrics_dict)

    train_y, train_x = get_nn_pred(args, model, device, train_loader)
    test_y, test_x = get_nn_pred(args, model, device, test_loader)
    train_x = train_x.float().detach()
    train_y = train_y.float().detach()
    test_x = test_x.float().detach()
    test_y = test_y.float().detach()

    y_pred = []
    y_var = []
    
    for i in range(train_x.shape[-1]):
        train_x_ = train_x[:, i]
        train_y_ = train_y[:, i]
        test_x_ = test_x[:, i]
        test_y_ = test_y[:, i]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = ExactGPModel(train_x_, train_y_, likelihood)

        gp_model.train()
        likelihood.train()
        
        gp_optimizer = optim.Adam(gp_model.parameters(), lr=args.lr, weight_decay=args.decay)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        observed_pred= dkl_process(gp_model, gp_optimizer, likelihood, mll, train_x_, train_y_, test_x_, test_y_, args.dkl_epochs, args)

        y_pred.append(observed_pred.mean.unsqueeze(1))
        y_var.append(torch.diagonal(observed_pred.covariance_matrix).unsqueeze(1))
        
    y_pred = torch.cat(y_pred, dim=1)
    y_var = torch.cat(y_var, dim=1)
    
    return y_pred, y_var, y_true

def dkl(model, train_loader, val_loader, test_loader, device, args):
    train_nn(args, model, device, train_loader)
    show_rep_snr(model, test_loader, device, args, tag='nn')
    y_true, y_pred = get_pred(args, model, device, test_loader)
    if args.task_type == 'cls':
        result_metrics_dict = classification_metrics(y_pred, y_true)
    elif args.task_type == 'reg':
        result_metrics_dict = regression_metrics(y_pred, None, y_true)
    print(result_metrics_dict)

    train_y, train_x = get_pred(args, model, device, train_loader)
    test_y, test_x = get_pred(args, model, device, test_loader)
    train_x = train_x.float().squeeze(1)
    train_y = train_y.float().squeeze(1)
    test_x = test_x.float().squeeze(1)
    test_y = test_y.float().squeeze(1)

    y_pred = []
    y_var = []
    
    #for i in range(train_x.shape[-1]):
        #train_x_ = train_x[:, i]
        #train_y_ = train_y[:, i]
        #test_x_ = test_x[:, i]
        #test_y_ = test_y[:, i]
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    gp_model.train()
    likelihood.train()
    
    gp_param_group = []
    #gp_param_group = get_parameters(model, gp_model, lr=args.lr*args.lr_scale, dkl_s=args.dkl_s, args=args)
    gp_param_group.append({"params": gp_model.parameters(), "lr":0.01})
    gp_optimizer = optim.Adam(gp_param_group, lr=args.lr, weight_decay=args.decay)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    observed_pred= dkl_process(gp_model, gp_optimizer, likelihood, mll, train_x, train_y, test_x, test_y, args.dkl_epochs, args)

    y_pred.append(observed_pred.mean.unsqueeze(1))
    y_var.append(torch.diagonal(observed_pred.covariance_matrix).unsqueeze(1))

    show_rep_snr(model, test_loader, device, args, tag='dkl')

    y_pred = torch.cat(y_pred, dim=1)
    y_var = torch.cat(y_var, dim=1)
    
    return y_pred, y_var, y_true


def old_svdkl(model, train_loader, val_loader, test_loader, device, args):
    train_nn(args, model, device, train_loader, test_loader)
    
    y_true, y_pred = get_pred(args, model, device, test_loader)

    result_metrics_dict = classification_metrics(y_pred, y_true)
    print(result_metrics_dict)

    # define DKL model
    #likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=args.num_tasks, num_classes=2)
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    grid_bounds=(-args.grid_bounds, args.grid_bounds)
    #GNN_DKL = GNN_SVDKL(args.num_layer, args.emb_dim, args.num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, dkl_s=args.dkl_s, grid_bounds=grid_bounds)
    #GNN_DKL.init_state(model)
    #GNN_DKL.nn = model

    is_model_equal(model, GNN_DKL.nn, device)
    
    if torch.cuda.is_available():
        GNN_DKL = GNN_DKL.to(device)
        likelihood = likelihood.to(device)
    GNN_DKL.eval()
    show_rep_snr(GNN_DKL.nn, test_loader, device, args, tag='nn')
    
    optimizer = optim.Adam(GNN_DKL.parameters(), lr=args.lr*args.dkl_scale, weight_decay=args.decay)
    #optimizer = optim.Adam(GNN_DKL.get_parameters(args.lr), lr=args.lr, weight_decay=args.decay)
    mll = gpytorch.mlls.VariationalELBO(likelihood, GNN_DKL.gp, num_data=len(train_loader.dataset))

    # train DKL
    dkl_epochs = args.dkl_epochs
    GNN_DKL.train()
    likelihood.train()
    for de in range(dkl_epochs):
        loss_log = []
        #for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            output = GNN_DKL(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            #y = batch.y.view(-1, args.num_tasks).to(torch.float64)
            y=batch.y.to(torch.float64)
            #print(y.shape)
            #y = (y+1)/2
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Calc loss and backprop gradients

            loss = -mll(output, y).mean()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())

        loss_mean = np.mean(loss_log)
        loss_var = np.var(loss_log)
        print(f'Iter {de+1}/{dkl_epochs} - Loss-mean: {loss_mean} - Loss-var: {loss_var}   lengthscale: %.3f ' % (
            GNN_DKL.gp.covar_module.base_kernel.lengthscale.item()
        ))
        if args.tsb:
            if de % 2 == 0:
                cls_train_stat(GNN_DKL, likelihood, test_loader, de, device, args)
    GNN_DKL.eval()
    likelihood.eval()
    show_rep_snr(GNN_DKL.nn, test_loader, device, args, tag='dkl')
    is_model_equal(model.gnn, GNN_DKL.nn.gnn, device)
    is_model_equal(model, GNN_DKL.nn, device)
    # test DKL
    y_pred = []
    y_var = []
    y_true = []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
            batch = batch.to(device)

            with torch.no_grad():
                pred = GNN_DKL(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            output = likelihood(pred)

            y_true.append(batch.y.view(pred.mean.shape))
            y_pred.append(pred.mean)
            y_var.append(torch.diagonal(pred.covariance_matrix).unsqueeze(1))

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_var = torch.cat(y_var, dim=0)
    
    return y_pred.cpu(), y_var.cpu(), y_true.cpu()

def svdkl(model, train_loader, val_loader, test_loader, device, args):
    train_nn(args, model, device, train_loader, test_loader)
    
    y_true, y_pred = get_pred(args, model, device, test_loader)

    result_metrics_dict = classification_metrics(y_pred, y_true)
    print(result_metrics_dict)

    # define DKL model
    GNN_DKL = SVDKL(args.num_layer, args.emb_dim, args.num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, device=args.device, dkl_s=args.dkl_s)

    # train and test for each task
    y_true = []
    y_pred = []
    y_var = []

    GNN_DKL.init_model(model, train_loader, device=device, type=args.ind_type, n_inducing_points=args.n_inducing_points)
    
    if torch.cuda.is_available():
        GNN_DKL = GNN_DKL.to(device)
        GNN_DKL.likelihood = GNN_DKL.likelihood.to(device)

    show_rep_snr(GNN_DKL.nn, test_loader, device, args, tag='nn')
    
    optimizer = optim.Adam(GNN_DKL.parameters(), lr=args.lr*args.dkl_scale, weight_decay=args.decay_dkl)
    #optimizer = optim.Adam(GNN_DKL.get_parameters(args.lr), lr=args.lr, weight_decay=args.decay)

    mll = gpytorch.mlls.VariationalELBO(GNN_DKL.likelihood, GNN_DKL.gp, num_data=len(train_loader.dataset))
    # train DKL
    dkl_epochs = args.dkl_epochs
    GNN_DKL.train()
    GNN_DKL.likelihood.train()
    for de in range(dkl_epochs):
        loss_log = []
        #for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            output = GNN_DKL(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y=batch.y.to(torch.float64)
            #y = (y+1)/2
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Calc loss and backprop gradients
            loss = -mll(output, y).mean()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())

        loss_mean = np.mean(loss_log)
        loss_var = np.var(loss_log)
        print(f'Iter {de+1}/{dkl_epochs} - Loss-mean: {loss_mean} - Loss-var: {loss_var}   lengthscale: %.3f ' % (
            GNN_DKL.gp.covar_module.base_kernel.lengthscale.item()
        ))
        if args.tsb:
            if de % 2 == 0:
                cls_train_stat(GNN_DKL, GNN_DKL.likelihood, test_loader, de, device, args)
    GNN_DKL.eval()
    GNN_DKL.likelihood.eval()
    show_rep_snr(GNN_DKL.nn, test_loader, device, args, tag='dkl')
    print('Is encoder kept:')
    is_model_equal(model.gnn, GNN_DKL.nn.gnn, device)
    print('Is prediction head kept:')
    is_model_equal(model, GNN_DKL.nn, device)
    # test DKL
    y_pred = []
    y_var = []
    y_true = []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
            batch = batch.to(device)

            with torch.no_grad():
                pred = GNN_DKL(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            output = GNN_DKL.likelihood(pred)

            y_true.append(batch.y.view(pred.mean.shape))
            y_pred.append(pred.mean)
            y_var.append(torch.diagonal(pred.covariance_matrix).unsqueeze(1))

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_var = torch.cat(y_var, dim=0)

    y_true = y_true.unsqueeze(1)
    y_pred = y_pred.unsqueeze(1)

    return y_pred.cpu(), y_var.cpu(), y_true.cpu()

def multi_single_svdkl(model, train_loader, val_loader, test_loader, device, args):
    train_nn(args, model, device, train_loader, test_loader)
    print(next(model.parameters()).device)
    y_true, y_pred = get_pred(args, model, device, test_loader)
    if args.task_type == 'cls':
        result_metrics_dict = classification_metrics(y_pred, y_true, note=f'{args.dataset}_none')
    else:
        result_metrics_dict = regression_metrics(y_pred, None, y_true)
    print(result_metrics_dict)

    # define DKL model
    GNN_DKL = SVDKL(args.num_layer, args.emb_dim, args.num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, device=args.device, dkl_s=args.dkl_s)

    # train and test for each task
    y_true = []
    y_pred = []
    y_var = []
    for i in range(args.num_tasks):
        GNN_DKL.init_model(model, train_loader, device=device, type=args.ind_type, n_inducing_points=args.n_inducing_points)
        
        if torch.cuda.is_available():
            GNN_DKL = GNN_DKL.to(device)
            GNN_DKL.likelihood = GNN_DKL.likelihood.to(device)
        
        if args.task_type == 'cls':
            show_rep_snr(GNN_DKL.nn, test_loader, device, args, tag='nn')
        
        optimizer = optim.Adam(GNN_DKL.parameters(), lr=args.lr*args.dkl_scale, weight_decay=args.decay_dkl)
        #optimizer = optim.Adam(GNN_DKL.get_parameters(args.lr), lr=args.lr, weight_decay=args.decay)

        mll = gpytorch.mlls.VariationalELBO(GNN_DKL.likelihood, GNN_DKL.gp, num_data=len(train_loader.dataset))
        #mll = gpytorch.mlls.PredictiveLogLikelihood(GNN_DKL.likelihood, GNN_DKL.gp, num_data=len(train_loader.dataset))
        # train DKL
        dkl_epochs = args.dkl_epochs
        GNN_DKL.train()
        GNN_DKL.likelihood.train()
        for de in range(dkl_epochs):
            loss_log = []
            #for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            for step, batch in enumerate(train_loader):
                batch = batch.to(device)
                output = GNN_DKL(batch.x, batch.edge_index, batch.edge_attr, batch.batch, i)
                y=batch.y.to(torch.float64).view(-1, args.num_tasks)[:, i]

                #y = (y+1)/2
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Calc loss and backprop gradients
                loss = -mll(output, y).mean()
                loss.backward()
                optimizer.step()
                loss_log.append(loss.item())

            loss_mean = np.mean(loss_log)
            loss_var = np.var(loss_log)
            print(f'Iter {de+1}/{dkl_epochs} - Loss-mean: {loss_mean} - Loss-var: {loss_var}   lengthscale: %.3f ' % (
                GNN_DKL.gp.covar_module.base_kernel.lengthscale.item()
            ))
            if args.tsb:
                if de % 2 == 0:
                    if args.task_type == 'cls':
                        cls_train_stat(GNN_DKL, GNN_DKL.likelihood, i, test_loader, de, device, args)
                    else:
                        reg_train_stat(GNN_DKL, GNN_DKL.likelihood, i, test_loader, de, device, args)
        GNN_DKL.eval()
        GNN_DKL.likelihood.eval()
        if args.task_type == 'cls':
            show_rep_snr(GNN_DKL.nn, test_loader, device, args, tag='dkl')
        print('Is encoder kept:')
        is_model_equal(model.gnn, GNN_DKL.nn.gnn, device)
        print('Is prediction head kept:')
        is_model_equal(model, GNN_DKL.nn, device)
        # test DKL
        y_pred_i = []
        y_var_i = []
        y_true_i = []
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
                batch = batch.to(device)

                with torch.no_grad():
                    pred = GNN_DKL(batch.x, batch.edge_index, batch.edge_attr, batch.batch, i)
                output = GNN_DKL.likelihood(pred)
                y=batch.y.view(-1, args.num_tasks)[:, i].view(-1, 1)
                y_true_i.append(y)
                y_pred_i.append(pred.mean)
                y_var_i.append(torch.diagonal(pred.covariance_matrix).unsqueeze(1))

        y_true_i = torch.cat(y_true_i, dim=0)
        y_pred_i = torch.cat(y_pred_i, dim=0)
        y_var_i = torch.cat(y_var_i, dim=0)

        y_true.append(y_true_i.view(-1, 1))
        y_pred.append(y_pred_i.view(-1, 1))
        y_var.append(y_var_i.view(-1, 1))

    y_true = torch.cat(y_true, dim=-1)
    y_pred = torch.cat(y_pred, dim=-1)
    y_var = torch.cat(y_var, dim=-1)

    return y_pred.cpu(), y_var.cpu(), y_true.cpu()

def embsvdkl(model, train_loader, val_loader, test_loader, device, args, fix_pretrain=False):
    if fix_pretrain:
        model.fix = True
    train_nn(args, model, device, train_loader, test_loader)
    
    y_true, y_pred = get_pred(args, model, device, test_loader)
    if args.task_type == 'cls':
        result_metrics_dict = classification_metrics(y_pred, y_true, note=f'{args.dataset}_none')
    else:
        result_metrics_dict = regression_metrics(y_pred, None, y_true)
    if args.task_type == 'cls':
        nn_acc = result_metrics_dict['roc-auc']['macro-avg']
    else:
        nn_acc = result_metrics_dict['rmse']['macro-avg']
    print(result_metrics_dict)

    # define DKL model
    GNN_DKL = embSVDKL(args.num_layer, args.emb_dim, args.proj_dim, args.num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, 
                       gnn_type = args.gnn_type, device=args.device, n_inducing_points=args.n_inducing_points, dkl_s=args.dkl_s)

    # train and test for each task
    y_true = []
    y_pred = []
    y_var = []
    for i in range(args.num_tasks):
        GNN_DKL.init_model(model, train_loader, device=device, type=args.ind_type)
        
        if torch.cuda.is_available():
            GNN_DKL = GNN_DKL.to(device)
            GNN_DKL.likelihood = GNN_DKL.likelihood.to(device)
        
        if args.task_type == 'cls':
            show_rep_snr(GNN_DKL.nn, test_loader, device, args, tag='nn')
        
        optimizer = optim.Adam(GNN_DKL.parameters(), lr=args.lr*args.dkl_scale, weight_decay=args.decay_dkl)
        #optimizer = optim.Adam(GNN_DKL.get_parameters(args.lr), lr=args.lr, weight_decay=args.decay)

        #mll = gpytorch.mlls.VariationalELBO(GNN_DKL.likelihood, GNN_DKL.gp, num_data=len(train_loader.dataset))
        mll = gpytorch.mlls.PredictiveLogLikelihood(GNN_DKL.likelihood, GNN_DKL.gp, num_data=len(train_loader.dataset))
        mll_pll = gpytorch.mlls.PredictiveLogLikelihood(GNN_DKL.likelihood, GNN_DKL.gp, num_data=len(train_loader.dataset), combine_terms=False)
        mll_ELBO = gpytorch.mlls.VariationalELBO(GNN_DKL.likelihood, GNN_DKL.gp, num_data=len(train_loader.dataset), combine_terms=False)
        # train DKL
        dkl_epochs = args.dkl_epochs
        GNN_DKL.train()
        GNN_DKL.likelihood.train()
        for de in range(dkl_epochs):
            loss_log = []
            log_likelihood_log = []
            kl_divergence_log = []
            log_prior_log = []
            var_log = []
            #for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            for step, batch in enumerate(train_loader):
                """
                if step > 0:
                    continue
                """
                batch = batch.to(device)
                output = GNN_DKL(batch.x, batch.edge_index, batch.edge_attr, batch.batch, i)
                y=batch.y.to(torch.float64).view(-1, args.num_tasks)[:, i]
                if args.task_type == 'cls':
                    y = (y+1)/2
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Calc loss and backprop gradients
                if args.ll == 'pll':
                    #loss = -mll_pll(output, y).mean()
                    log_likelihood, kl_divergence, log_prior = mll_pll(output, y)
                    loss = - (log_likelihood - kl_divergence + log_prior).mean()
                else:
                    #loss = -mll_ELBO(output, y).mean()
                    log_likelihood, kl_divergence, log_prior = mll_ELBO(output, y)
                    loss = - (log_likelihood - kl_divergence + log_prior).mean()
                
                loss.backward()
                optimizer.step()
                loss_log.append(loss.item())
                log_likelihood_log.append(log_likelihood.item())
                kl_divergence_log.append(kl_divergence.item())
                log_prior_log.append(log_prior.item())
                var = torch.diagonal(output.covariance_matrix).mean()
                var_log.append(var.item())

            loss_mean = np.mean(loss_log)
            loss_var = np.var(loss_log)
            log_like_mean = np.mean(log_likelihood_log)
            kl_mean = np.mean(kl_divergence_log)
            log_prior_mean = np.mean(log_prior_log)
            var_mean = np.mean(var_log)
            print(f'Iter {de+1}/{dkl_epochs} - Loss-mean: {loss_mean} - Loss-var: {loss_var}   lengthscale: %.3f ' % (
                GNN_DKL.gp.covar_module.base_kernel.lengthscale.item()
            ))
            print(f'log_likelihood: {log_like_mean}, kl: {kl_mean}, log_prior: {log_prior_mean}, var: {var_mean}')
            #print(f'Iter {de+1}/{dkl_epochs} - Loss-mean: {loss_mean} - Loss-var: {loss_var}')
            if args.tsb:
                ls = GNN_DKL.gp.covar_module.base_kernel.lengthscale.item()
                if de % 2 == 0:
                    if args.task_type == 'cls':
                        cls_train_stat(GNN_DKL, GNN_DKL.likelihood, ls, i, test_loader, de, device, args)
                    else:
                        reg_train_stat(GNN_DKL, GNN_DKL.likelihood, ls, i, test_loader, de, device, args)
        if args.extra_pll:
            extra_dkl_epochs = dkl_epochs + int(dkl_epochs/2)
            for de in range(dkl_epochs, extra_dkl_epochs):
                loss_log = []
                log_likelihood_log = []
                kl_divergence_log = []
                log_prior_log = []
                var_log = []
                #for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
                for step, batch in enumerate(train_loader):
                    batch = batch.to(device)
                    output = GNN_DKL(batch.x, batch.edge_index, batch.edge_attr, batch.batch, i)
                    y=batch.y.to(torch.float64).view(-1, args.num_tasks)[:, i]
                    if args.task_type == 'cls':
                        y = (y+1)/2
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Calc loss and backprop gradients
                    #loss = -mll_pll(output, y).mean()
                    log_likelihood, kl_divergence, log_prior = mll_pll(output, y)
                    loss = - (log_likelihood - kl_divergence + log_prior ).mean()
                    loss.backward()
                    optimizer.step()
                    loss_log.append(loss.item())
                    log_likelihood_log.append(log_likelihood.item())
                    kl_divergence_log.append(kl_divergence.item())
                    log_prior_log.append(log_prior.item())
                    var = torch.diagonal(output.covariance_matrix).mean()
                    var_log.append(var.item())

                loss_mean = np.mean(loss_log)
                loss_var = np.var(loss_log)
                loss_var = np.var(loss_log)
                log_like_mean = np.mean(log_likelihood_log)
                kl_mean = np.mean(kl_divergence_log)
                log_prior_mean = np.mean(log_prior_log)
                var_mean = np.mean(var_log)
                print(f'Iter {de+1}/{extra_dkl_epochs} - Loss-mean: {loss_mean} - Loss-var: {loss_var}   lengthscale: %.3f ' % (
                    GNN_DKL.gp.covar_module.base_kernel.lengthscale.item()
                ))
                print(f'log_likelihood: {log_like_mean}, kl: {kl_mean}, log_prior: {log_prior_mean}, var: {var_mean}')
                if args.tsb:
                    if de % 2 == 0:
                        if args.task_type == 'cls':
                            cls_train_stat(GNN_DKL, GNN_DKL.likelihood, ls, i, test_loader, de, device, args)
                        else:
                            reg_train_stat(GNN_DKL, GNN_DKL.likelihood, ls, i, test_loader, de, device, args)
            
        GNN_DKL.eval()
        GNN_DKL.likelihood.eval()
        if args.task_type == 'cls':
            show_rep_snr(GNN_DKL.nn, test_loader, device, args, tag='dkl')
        print('Is encoder kept:')
        is_model_equal(model.gnn, GNN_DKL.nn.gnn, device)
        print('Is prediction head kept:')
        is_model_equal(model, GNN_DKL.nn, device)
        # test DKL
        y_pred_i = []
        y_var_i = []
        y_true_i = []
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
                batch = batch.to(device)

                with torch.no_grad():
                    pred = GNN_DKL(batch.x, batch.edge_index, batch.edge_attr, batch.batch, i)
                output = GNN_DKL.likelihood(pred)
                y=batch.y.view(-1, args.num_tasks)[:, i].view(-1, 1)
                y_true_i.append(y)
                y_pred_i.append(pred.mean)
                y_var_i.append(torch.diagonal(pred.covariance_matrix).unsqueeze(1))

        y_true_i = torch.cat(y_true_i, dim=0)
        y_pred_i = torch.cat(y_pred_i, dim=0)
        y_var_i = torch.cat(y_var_i, dim=0)

        y_true.append(y_true_i.view(-1, 1))
        y_pred.append(y_pred_i.view(-1, 1))
        y_var.append(y_var_i.view(-1, 1))

        """
        if args.save_kernel:
            save_kernel(pred.covariance_matrix, args.dataset, str(i), args)
            save_model(GNN_DKL, args.dataset, str(i), args)
        """

    y_true = torch.cat(y_true, dim=-1)
    y_pred = torch.cat(y_pred, dim=-1)
    y_var = torch.cat(y_var, dim=-1)

    return y_pred.cpu(), y_var.cpu(), y_true.cpu(), GNN_DKL, nn_acc