import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import math
import gpytorch
from gpytorch.models import ApproximateGP, GP
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  DeltaVariationalDistribution,
                                VariationalStrategy,
                                 AdditiveGridInterpolationVariationalStrategy)
from gpytorch.constraints import GreaterThan
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, DirichletClassificationLikelihood, LaplaceLikelihood
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from models.model import GNN_graphpred, GNN_proj_graphpred
from models.GP import Standard_GPModel, Multi_GPModel
from base.train.utils import calc_inducing_points, calc_emb_inducing_points

from copy import deepcopy
from tqdm import tqdm

"""
nn contains encoder and lin_pred
"""
class SVDKL(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_tasks, JK, drop_ratio, graph_pooling, gnn_type, device, dkl_s='s1'):
        super(SVDKL, self).__init__()
        self.num_tasks = num_tasks
        self.device = device
        self.nn = GNN_graphpred(num_layer, emb_dim, num_tasks, JK = JK, drop_ratio = drop_ratio, graph_pooling = graph_pooling, gnn_type = gnn_type)

        self.dkl_s = dkl_s

    def copy_nn_state(self, ori_model):
        self.nn.gnn.load_state_dict(ori_model.gnn.state_dict())
        self.nn.graph_pred_linear.load_state_dict(ori_model.graph_pred_linear.state_dict())

    def init_model(self, encoder, train_loader, device, task_index=0, type='kmeans', n_inducing_points=64):
        device = self.device
        self.copy_nn_state(encoder)
        inducing_points, initial_lengthscale = calc_inducing_points(
            model=encoder, 
            train_loader=train_loader,
            task_index=task_index,
            type=type, 
            n_inducing_points=n_inducing_points,
            device=device
        )
        print(inducing_points.shape)
        kernel = gpytorch.kernels.RBFKernel()
        kernel.lengthscale = initial_lengthscale.to(device) * torch.ones_like(kernel.lengthscale, device=device)
        covar_module = gpytorch.kernels.ScaleKernel(kernel)
        gp = Standard_GPModel(inducing_points, covar_module).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        #likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        #likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=1)

        self.gp = gp
        self.likelihood = likelihood


    def forward(self, h, edge_index, edge_attr, batch, task_index=0):

        if self.dkl_s == 's1':
            x = self.nn.encoder(h, edge_index, edge_attr, batch)
            x = self.nn.lin_pred(x)
        elif self.dkl_s == 's2':
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
            x = self.nn.lin_pred(x)
        else:
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
                x = self.nn.lin_pred(x).detach()

        # task index
        #print(x.shape)
        x = x[:, task_index].view(-1, 1)
        #x = torch.sigmoid(x)
        pred = self.gp(x)
        return pred

    def encoder(self, h, edge_index, edge_attr, batch, fix=False):
        if fix:
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
        else:
            x = self.nn.encoder(h, edge_index, edge_attr, batch)
        
        return x

    def pred(self, h, edge_index, edge_attr, batch, fix=False):
        if fix:
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
                x = self.nn.lin_pred(x)
        else:
            x = self.nn.encoder(h, edge_index, edge_attr, batch)
            x = self.nn.lin_pred(x)
        
        return x

    def get_parameters(self, lr):
        gp_param_group = []
        dkl_s = self.dkl_s
        if dkl_s=='s1':
            gp_param_group.append({"params": self.nn.gnn.parameters()})
            if self.nn.graph_pooling == "attention":
                gp_param_group.append({"params": self.nn.pool.parameters(), "lr":lr})
            gp_param_group.append({"params": self.nn.graph_pred_linear.parameters(), "lr":lr})
        elif dkl_s=='s2':
            gp_param_group.append({"params": self.nn.graph_pred_linear.parameters(), "lr":lr})

        gp_param_group.append({"params": self.gp.parameters(), "lr":0.01})

        return gp_param_group

"""
nn contains encoder and lin_pred
"""
class MultiSVDKL(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_tasks, JK, drop_ratio, graph_pooling, gnn_type, device, dkl_s='s1'):
        super(MultiSVDKL, self).__init__()
        self.num_tasks = num_tasks
        self.device = device
        self.nn = GNN_graphpred(num_layer, emb_dim, num_tasks, JK = JK, drop_ratio = drop_ratio, graph_pooling = graph_pooling, gnn_type = gnn_type)

        self.dkl_s = dkl_s

    def copy_nn_state(self, ori_model):
        self.nn.gnn.load_state_dict(ori_model.gnn.state_dict())
        self.nn.graph_pred_linear.load_state_dict(ori_model.graph_pred_linear.state_dict())

    def init_model(self, encoder, train_loader, num_tasks, device, type='kmeans', n_inducing_points=64):
        device = self.device
        self.copy_nn_state(encoder)
        inducing_points, initial_lengthscale = calc_inducing_points(
            model=encoder, 
            train_loader=train_loader,
            num_tasks=num_tasks,
            type=type, 
            n_inducing_points=n_inducing_points,
            device=device
        )
        print(inducing_points.shape)
        kernel = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks]))
        kernel.lengthscale = initial_lengthscale.to(device) * torch.ones_like(kernel.lengthscale, device=device)
        covar_module = gpytorch.kernels.ScaleKernel(kernel, batch_shape=torch.Size([num_tasks]))
        #gp = Standard_GPModel(inducing_points, covar_module).to(device)
        gp = Multi_GPModel(inducing_points, covar_module, num_tasks).to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
        #likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        #likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        #likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=1)

        self.gp = gp
        self.likelihood = likelihood


    def forward(self, h, edge_index, edge_attr, batch):

        if self.dkl_s == 's1':
            x = self.nn.encoder(h, edge_index, edge_attr, batch)
            x = self.nn.lin_pred(x)
        elif self.dkl_s == 's2':
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
            x = self.nn.lin_pred(x)
        else:
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
                x = self.nn.lin_pred(x).detach()

        pred = self.gp(x)
        return pred

    def encoder(self, h, edge_index, edge_attr, batch, fix=False):
        if fix:
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
        else:
            x = self.nn.encoder(h, edge_index, edge_attr, batch)
        
        return x

    def pred(self, h, edge_index, edge_attr, batch, fix=False):
        if fix:
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
                x = self.nn.lin_pred(x)
        else:
            x = self.nn.encoder(h, edge_index, edge_attr, batch)
            x = self.nn.lin_pred(x)
        
        return x

    def get_parameters(self, lr):
        gp_param_group = []
        dkl_s = self.dkl_s
        if dkl_s=='s1':
            gp_param_group.append({"params": self.nn.gnn.parameters()})
            if self.nn.graph_pooling == "attention":
                gp_param_group.append({"params": self.nn.pool.parameters(), "lr":lr})
            gp_param_group.append({"params": self.nn.graph_pred_linear.parameters(), "lr":lr})
        elif dkl_s=='s2':
            gp_param_group.append({"params": self.nn.graph_pred_linear.parameters(), "lr":lr})

        gp_param_group.append({"params": self.gp.parameters(), "lr":0.01})

        return gp_param_group

"""
use embedding
"""
class embSVDKL(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, proj_dim, num_tasks, JK, drop_ratio, graph_pooling, gnn_type, device, n_inducing_points=64, dkl_s='s1'):
        super(embSVDKL, self).__init__()
        self.num_tasks = num_tasks
        self.device = device
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.nn = GNN_proj_graphpred(num_layer, emb_dim, proj_dim, num_tasks, JK = JK, drop_ratio = drop_ratio, graph_pooling = graph_pooling, gnn_type = gnn_type)
        #self.nn = GNN_graphpred(num_layer, emb_dim, num_tasks, JK = JK, drop_ratio = drop_ratio, graph_pooling = graph_pooling, gnn_type = gnn_type)

        self.n_inducing_points = n_inducing_points
        self.dkl_s = dkl_s

        kernel = gpytorch.kernels.RBFKernel()
        covar_module = gpytorch.kernels.ScaleKernel(kernel)
        gp = Standard_GPModel(torch.rand(self.n_inducing_points, proj_dim), covar_module).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.gp = gp
        self.likelihood = likelihood

    def copy_nn_state(self, ori_model):
        self.nn.load_state_dict(ori_model.state_dict())
        #self.nn.gnn.load_state_dict(ori_model.gnn.state_dict())
        #self.nn.graph_pred_linear.load_state_dict(ori_model.graph_pred_linear.state_dict())

    def init_model(self, encoder, train_loader, device, task_index=0, type='kmeans'):
        device = self.device
        self.copy_nn_state(encoder)
        inducing_points, initial_lengthscale = calc_emb_inducing_points(
            model=encoder, 
            train_loader=train_loader,
            task_index=task_index,
            type=type, 
            n_inducing_points=self.n_inducing_points,
            device=device
        )
        print(inducing_points.shape)
        kernel = gpytorch.kernels.RBFKernel()
        kernel.lengthscale = initial_lengthscale.to(device) * torch.ones_like(kernel.lengthscale, device=device)
        covar_module = gpytorch.kernels.ScaleKernel(kernel)
        gp = Standard_GPModel(inducing_points, covar_module).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        self.gp = gp
        self.likelihood = likelihood

    def forward(self, h, edge_index, edge_attr, batch, task_index=0):
        #x = self.nn.encoder(h, edge_index, edge_attr, batch)
        
        self.nn.eval()
        with torch.no_grad():
            x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
        
        """
        self.nn.eval()
        with torch.no_grad():
            x = self.nn.enc_wo_proj(h, edge_index, edge_attr, batch).detach()
        self.nn.train()
        x = self.nn.proj(x)
        """
        pred = self.gp(x)
        return pred

    def encoder(self, h, edge_index, edge_attr, batch, fix=False):
        if fix:
            self.nn.eval()
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
        else:
            x = self.nn.encoder(h, edge_index, edge_attr, batch)
        
        return x
    
    def enc_wo_proj(self, h, edge_index, edge_attr, batch, fix=False):
        with torch.no_grad():
            x = self.nn.enc_wo_proj(h, edge_index, edge_attr, batch).detach()
        return x
    
    def proj_from_enc(self, emb):
        return self.nn.proj_from_enc(emb)

    def pred(self, h, edge_index, edge_attr, batch, fix=False):
        if fix:
            self.nn.eval()
            with torch.no_grad():
                x = self.nn.encoder(h, edge_index, edge_attr, batch).detach()
                x = self.nn.lin_pred(x)
        else:
            x = self.nn.encoder(h, edge_index, edge_attr, batch)
            x = self.nn.lin_pred(x)
        
        return x
