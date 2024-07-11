import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import gpytorch
from sklearn.metrics import roc_auc_score

from models.GP import GPModel, ExactGPModel, SVGPModel
from models.model import GNN_DKL
from base.train.utils import cls_train, reg_train, get_pred
from base.train.metrics import classification_metrics, regression_metrics
from base.repsentation.rep_analysis import *

from tqdm import tqdm

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

def train_nn(args, model, device, train_loader):
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    model_file = f'./models_finetune/{args.dataset}.pth'
    if args.load:
        model.load_state_dict(torch.load(model_file))
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

        with torch.no_grad():
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

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = GNN_DKL(args.num_layer, args.emb_dim, args.num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    gp_model.copy_state(model)

    if torch.cuda.is_available():
        gp_model = gp_model.cuda()
        likelihood = likelihood.cuda()

    gp_model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': gp_model.gnn.parameters()},
        {'params': gp_model.graph_pred_linear.parameters()},
        {'params': gp_model.covar_module.parameters()},
        {'params': gp_model.mean_module.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch = batch.to(device)
        output = gp_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(output.shape).to(torch.float64)

        optimizer.zero_grad()

        # Calc loss and backprop derivatives
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

    gp_model.eval()
    likelihood.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = gp_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_pred.append(pred.mean.unsqueeze(1))
        y_var.append(torch.diagonal(pred.covariance_matrix).unsqueeze(1))

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    show_rep_snr(gp_model, test_loader, device, args, tag='dkl')

    y_pred = torch.cat(y_pred, dim=1)
    y_var = torch.cat(y_var, dim=1)
    
    return y_pred, y_var, y_true

def svdkl(model, train_loader, val_loader, test_loader, device, args):
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    for epoch in range(args.epochs):
        if args.task_type == 'cls':
            cls_train(args, model, device, train_loader, optimizer)
        elif args.task_type == 'reg':
            reg_train(args, model, device, train_loader, optimizer)
    y_true, y_pred = get_nn_pred(args, model, device, test_loader)
    result_metrics_dict = classification_metrics(y_pred, y_true)
    print(result_metrics_dict)

    train_y, train_x = get_nn_pred(args, model, device, train_loader)
    test_y, test_x = get_nn_pred(args, model, device, test_loader)
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()

    print(train_x.shape)
    y_pred = []
    y_var = []
    for i in range(train_x.shape[-1]):
        train_x_ = train_x[:, i]
        train_y_ = train_y[:, i]
        test_x_ = test_x[:, i]
        test_y_ = test_y[:, i]
        inducing_points = train_x_[:args.num_inducing_points]

        gp_model = SVGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        """
        if torch.cuda.is_available():
            gp_model = gp_model.cuda()
            likelihood = likelihood.cuda()
        """
        model.train()
        gp_model.train()
        likelihood.train()

        # Use the adam optimizer
        #model_param_group.append({"params": gp_model.parameters(), "lr":0.01})
        gp_param_group = get_parameters(model, gp_model, lr=args.lr*args.lr_scale, dkl_s=args.dkl_s, args=args)
        gp_param_group.append({'params': likelihood.parameters()})
        """
        gp_optimizer = torch.optim.Adam([
            {"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale},
            {'params': gp_model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)
        """
        gp_optimizer = torch.optim.Adam(gp_param_group, lr=0.01)
        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=train_y_.size(0))

        gp_model.train()
        likelihood.train()
        dkl_epochs = args.dkl_epochs

        for i in range(dkl_epochs):
            # Zero gradients from previous iteration
            gp_optimizer.zero_grad()
            # Output from model
            output = gp_model(train_x_)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y_).mean()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
                i + 1, dkl_epochs, loss.item(),
                gp_model.covar_module.base_kernel.lengthscale.item()
            ))
            gp_optimizer.step()
        gp_model.eval()
        likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #test_x_ = torch.linspace(0, 1, 51).cuda()
            #print(test_x_)
            gp_model = gp_model.to('cpu')
            test_x_ = test_x_.to('cpu')
            observed_pred = likelihood(gp_model(test_x_))
        
        y_pred.append(observed_pred.mean.unsqueeze(1))
        y_var.append(torch.diagonal(observed_pred.covariance_matrix).unsqueeze(1))

    y_pred = torch.cat(y_pred, dim=1)
    y_var = torch.cat(y_var, dim=1)

    return y_pred, y_var, y_true