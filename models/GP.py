import torch
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
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from models.model import GNN_graphpred
from base.train.utils import calc_inducing_points

from copy import deepcopy

class GPModel(ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SVGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SVGP(GP):
    def __init__(self, feature_dim, label_dim, n_inducing):
        super(SVGP, self).__init__()
        noise_constraint = GreaterThan(1e-4)
        self.likelihood = GaussianLikelihood(
            batch_shape=torch.Size([label_dim]),
            noise_constraint=noise_constraint
        )
        self.mean_module = ConstantMean(batch_shape=torch.Size([label_dim]))
        base_kernel = RBFKernel(
            batch_shape=torch.Size([label_dim]),
            ard_num_dims=feature_dim
        )
        self.covar_module = ScaleKernel(base_kernel, batch_shape=torch.Size([label_dim]))

        variational_dist = MeanFieldVariationalDistribution(
            num_inducing_points=n_inducing,
            batch_shape=torch.Size([label_dim])
        )
        inducing_points = torch.randn(n_inducing, feature_dim)
        self.variational_strategy = VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True
        )

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): [n x feature_dim]
        Returns:
            GPyTorch MultivariateNormal distribution
        """
        mean = self.mean_module(features)
        covar = self.covar_module(features)
        return MultivariateNormal(mean, covar)

    def predict(self, np_inputs, latent=False):
        """
        Args:
            np_inputs (np.array): [n x input_dim]
            latent (bool): if True, predict latent function values (rather than label values)
        Returns:
            mean (np.array): [n x label_dim]
            var (np.array): [n x label_dim]
        """
        inputs = torch.tensor(np_inputs, dtype=torch.get_default_dtype())
        with torch.no_grad():
            pred_dist = self(inputs) if latent else self.likelihood(self(inputs))
        mean = pred_dist.mean * self.label_std.view(self.label_dim, 1) + self.label_mean.view(self.label_dim, 1)
        var = pred_dist.variance * self.label_std.pow(2).view(self.label_dim, 1)
        return mean.t().cpu().numpy(), var.t().cpu().numpy()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        #self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim=1, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )
        
        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )


        super().__init__(variational_strategy)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        """
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        """
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

"""
class GNN_SVDKL(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_tasks, JK, drop_ratio, graph_pooling, gnn_type, grid_bounds=(-1000., 1000.), grid_size=512, dkl_s='s1'):
        super(GNN_SVDKL, self).__init__()
        self.num_tasks = num_tasks
        self.nn = GNN_graphpred(num_layer, emb_dim, num_tasks, JK = JK, drop_ratio = drop_ratio, graph_pooling = graph_pooling, gnn_type = gnn_type)
        self.gp = GaussianProcessLayer(num_dim=1, grid_bounds=grid_bounds, grid_size=grid_size)
        self.grid_bounds = grid_bounds
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

        self.dkl_s = dkl_s

    def forward(self, h, edge_index, edge_attr, batch):
        
        if self.dkl_s == 's1':
            x = self.nn.get_rep(h, edge_index, edge_attr, batch)
            x = self.nn.get_pred(x)
        elif self.dkl_s == 's2':
            with torch.no_grad():
                x = self.nn.get_rep(h, edge_index, edge_attr, batch).detach()
            x = self.nn.get_pred(x)
        else:
            with torch.no_grad():
                x = self.nn.get_rep(h, edge_index, edge_attr, batch).detach()
                x = self.nn.get_pred(x).detach()

        #x = self.scale_to_bounds(x)
        pred = self.gp(x)
        return pred

    def copy_state(self, ori_model):
        self.nn.gnn.load_state_dict(ori_model.gnn.state_dict())
        self.nn.graph_pred_linear.load_state_dict(ori_model.graph_pred_linear.state_dict())
    
    def init_state(self, ori_model):
        self.copy_state(ori_model)
        #self.gp = GaussianProcessLayer()
 
    def fix_param(self):
        if self.dkl_s == 's2':
            for p in self.nn.gnn.parameters():
                p.requires_grad = False
        elif self.dkl_s == 's3':
            for p in self.nn.parameters():
                p.requires_grad = False


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

class Standard_GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, covar_module):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(Standard_GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x, batch=None):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Multi_GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, covar_module, num_tasks):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([num_tasks]))
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ), num_tasks=num_tasks)
        super(Multi_GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = covar_module

    def forward(self, x, batch=None):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
