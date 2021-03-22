import time
import logging
import numpy as np
import emcee
import os 
import sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import optimize
from utils.utils import preprocessing
from models.configs import configs
from pybnn.base_model import BaseModel
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from pybnn.bayesian_linear_regression import BayesianLinearRegression, Prior
from gin.models.mlp import MLP


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, normalize=False):
        super(Net, self).__init__()
        self.normalize = normalize
        # GIN
        self.num_layers = num_hops
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.fc = nn.Linear(self.hidden_dim, self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 1)


    def _encoder(self, ops, adj):
        batch_size, node_num, opt_num = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj, x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        x = self.fc(x)
        return x

    def forward(self, ops, adj):
        ops = ops.cuda()
        adj = adj.cuda()
        x = self._encoder(ops, adj)
        x = torch.tanh(torch.mean(x, dim=1).squeeze(1))
        return self.out(x).cpu()

    def basis_funcs(self, ops, adj):
        ops = ops.cuda()
        adj = adj.cuda()
        x = self._encoder(ops, adj)
        x = torch.tanh(torch.mean(x, dim=1).squeeze(1))
        return x.cpu()


class DNGO(BaseModel):
    def __init__(self, input_dim=5, hidden_dim=128, latent_dim=16, num_hops=5, num_mlp_layers=2,
                 batch_size=10, num_epochs=100, learning_rate=1e-2,
                 adapt_epoch=5000, alpha=1.0, beta=1000, prior=None, do_mcmc=True,
                 n_hypers=20, chain_length=2000, burnin_steps=2000,
                 normalize_input=False, normalize_output=False, rng=None):

        """
        Deep Networks for Global Optimization [1]. This module performs
        Bayesian Linear Regression with basis function extracted from a
        feed forward neural network.

        [1] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish,
            N. Sundaram, M.~M.~A. Patwary, Prabhat, R.~P. Adams
            Scalable Bayesian Optimization Using Deep Neural Networks
            Proc. of ICML'15

        Parameters
        ----------
        batch_size: int
            Batch size for training the neural network
        num_epochs: int
            Number of epochs for training
        learning_rate: float
            Initial learning rate for Adam
        adapt_epoch: int
            Defines after how many epochs the learning rate will be decayed by a factor 10
        alpha: float
            Hyperparameter of the Bayesian linear regression
        beta: float
            Hyperparameter of the Bayesian linear regression
        prior: Prior object
            Prior for alpa and beta. If set to None the default prior is used
        do_mcmc: bool
            If set to true different values for alpha and beta are sampled via MCMC from the marginal log likelihood
            Otherwise the marginal log likehood is optimized with scipy fmin function
        n_hypers : int
            Number of samples for alpha and beta
        chain_length : int
            The chain length of the MCMC sampler
        burnin_steps: int
            The number of burnin steps before the sampling procedure starts
        normalize_output : bool
            Zero mean unit variance normalization of the output values
        normalize_input : bool
            Zero mean unit variance normalization of the input values
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.X = None
        self.y = None
        self.network = None
        self.alpha = alpha
        self.beta = beta
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        # MCMC hyperparameters
        self.do_mcmc = do_mcmc
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        if prior is None:
            self.prior = Prior(rng=self.rng)
        else:
            self.prior = prior

        # Network hyper parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_hops = num_hops
        self.num_mlp_layers = num_mlp_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

        self.adapt_epoch = adapt_epoch
        self.network = Net(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, num_hops=self.num_hops,
                           num_mlp_layers=self.num_mlp_layers, normalize=False).cuda()
        self.encoder = None
        self.models = []
        self.hypers = None

    def train(self, X_adj, X_ops, y, do_optimize=True):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters are used.

        """
        start_time = time.time()

        self.X_adj = X_adj
        self.X_ops = X_ops

        # Normalize ouputs
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y


        self.y = self.y[:, None]

        # Check if we have enough points to create a minibatch otherwise use all data points
        if self.X_adj.shape[0] <= self.batch_size:
            batch_size = self.X_adj.shape[0]
        else:
            batch_size = self.batch_size

        # Create the neural network
        features = X_ops.shape[1]



        optimizer = optim.Adam(self.network.parameters(),
                               lr=self.init_learning_rate)

        # Start training
        lc = np.zeros([self.num_epochs])
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.X_adj, self.X_ops, self.y,
                                                  batch_size, shuffle=True):
                inputs_adj = torch.Tensor(batch[0])
                inputs_ops = torch.Tensor(batch[1])
                targets = torch.Tensor(batch[2])

                optimizer.zero_grad()
                output = self.network(inputs_ops, inputs_adj)
                loss = torch.nn.functional.mse_loss(output, targets)
                loss.backward()
                optimizer.step()

                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            logging.debug("Epoch {} of {}".format(epoch + 1, self.num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
            #print("Training loss:\t\t{:.5g}".format(train_err / train_batches))

        # Design matrix
        self.Theta = self.network.basis_funcs(torch.Tensor(self.X_ops), torch.Tensor(self.X_adj)).data.numpy()

        if do_optimize:
            if self.do_mcmc:
                self.sampler = emcee.EnsembleSampler(self.n_hypers, 2,
                                                     self.marginal_log_likelihood)

                # Do a burn-in in the first iteration
                if not self.burned:
                    # Initialize the walkers by sampling from the prior
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                    # Run MCMC sampling
                    self.p0, _, _ = self.sampler.run_mcmc(self.p0,
                                                          self.burnin_steps,
                                                          rstate0=self.rng)

                    self.burned = True

                # Start sampling
                pos, _, _ = self.sampler.run_mcmc(self.p0,
                                                  self.chain_length,
                                                  rstate0=self.rng)

                # Save the current position, it will be the startpoint in
                # the next iteration
                self.p0 = pos

                # Take the last samples from each walker set them back on a linear scale
                linear_theta = np.exp(self.sampler.chain[:, -1])
                self.hypers = linear_theta
                self.hypers[:, 1] = 1 / self.hypers[:, 1]
            else:
                # Optimize hyperparameters of the Bayesian linear regression
                p0 = self.prior.sample_from_prior(n_samples=1)
                res = optimize.fmin(self.negative_mll, p0)
                self.hypers = [[np.exp(res[0]), 1 / np.exp(res[1])]]
        else:

            self.hypers = [[self.alpha, self.beta]]

        logging.info("Hypers: %s" % self.hypers)
        self.models = []
        for sample in self.hypers:
            # Instantiate a model for each hyperparameter configuration
            model = BayesianLinearRegression(alpha=sample[0],
                                             beta=sample[1],
                                             basis_func=None)
            model.train(self.Theta, self.y[:, 0], do_optimize=False)

            self.models.append(model)

    def marginal_log_likelihood(self, theta):
        """
        Log likelihood of the data marginalised over the weights w. See chapter 3.5 of
        the book by Bishop of an derivation.

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            lnlikelihood + prior
        """
        if np.any(theta == np.inf):
            return -np.inf

        if np.any((-10 > theta) + (theta > 10)):
            return -np.inf

        alpha = np.exp(theta[0])
        beta = 1 / np.exp(theta[1])

        D = self.Theta.shape[1]
        N = self.Theta.shape[0]

        K = beta * np.dot(self.Theta.T, self.Theta)
        K += np.eye(self.Theta.shape[1]) * alpha
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.linalg.LinAlgError:
             K_inv = np.linalg.inv(K + np.random.rand(K.shape[0], K.shape[1]) * 1e-8)

        m = beta * np.dot(K_inv, self.Theta.T)
        m = np.dot(m, self.y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.Theta, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(K) + 1e-10)

        if np.any(np.isnan(mll)):
            return -1e25
        return mll

    def negative_mll(self, theta):
        """
        Returns the negative marginal log likelihood (for optimizing it with scipy).

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            negative lnlikelihood + prior
        """
        nll = -self.marginal_log_likelihood(theta)
        return nll

    def iterate_minibatches(self, inputs_adj, inputs_ops, targets, batchsize, shuffle=False):
        assert inputs_adj.shape[0] == targets.shape[0], \
            "The number of training points is not the same"
        if shuffle:
            indices_adj = np.arange(inputs_adj.shape[0])
            self.rng.shuffle(indices_adj)
        for start_idx in range(0, inputs_adj.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices_adj[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs_adj[excerpt], inputs_ops[excerpt], targets[excerpt]

    def predict(self, X_ops_test, X_adj_test):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """
        X_ops_ = X_ops_test
        X_adj_ = X_adj_test

        # Get features from the net
        theta = self.network.basis_funcs(torch.Tensor(X_ops_), torch.Tensor(X_adj_)).data.numpy()

        # Marginalise predictions over hyperparameters of the BLR
        mu = np.zeros([len(self.models), X_ops_test.shape[0]])
        var = np.zeros([len(self.models), X_ops_test.shape[0]])

        for i, m in enumerate(self.models):
            mu[i], var[i] = m.predict(theta)

        # See the algorithm runtime prediction paper by Hutter et al
        # for the derivation of the total variance
        m = np.mean(mu, axis=0)
        v = np.mean(mu ** 2 + var, axis=0) - m ** 2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        return m, v

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """

        inc, inc_value = super(DNGO, self).get_incumbent()
        if self.normalize_input:
            inc = zero_mean_unit_var_denormalization(inc, self.X_mean, self.X_std)

        if self.normalize_output:
            inc_value = zero_mean_unit_var_denormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
