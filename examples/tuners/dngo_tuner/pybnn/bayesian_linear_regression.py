import emcee
import logging
import numpy as np

from scipy import optimize
from scipy import stats

from pybnn.base_model import BaseModel


def linear_basis_func(x):
    return np.append(x, np.ones([x.shape[0], 1]), axis=1)


def quadratic_basis_func(x):
    x = np.append(x ** 2, x, axis=1)
    return np.append(x, np.ones([x.shape[0], 1]), axis=1)


logger = logging.getLogger(__name__)


class Prior(object):

    def __init__(self, rng=None):
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

    def lnprob(self, theta):
        """
        Compute the log probability for theta = [log alpha, log beta]
        :param theta:
        :return: log p(theta)
        """
        lp = 0
        lp += stats.norm.pdf(theta[0], loc=0, scale=1)  # log alpha
        lp += stats.norm.pdf(theta[1], loc=0, scale=1)  # log sigma^2

        return lp

    def sample_from_prior(self, n_samples):
        p0 = np.zeros([n_samples, 2])

        # Log alpha
        p0[:, 0] = self.rng.normal(loc=0,
                                   scale=1,
                                   size=n_samples)

        # Log sigma^2
        p0[:, 1] = self.rng.normal(loc=-3,
                                   scale=1,
                                   size=n_samples)
        return p0


class BayesianLinearRegression(BaseModel):

    def __init__(self, alpha=1, beta=1000, basis_func=linear_basis_func,
                 prior=None, do_mcmc=True, n_hypers=20, chain_length=2000,
                 burnin_steps=2000, rng=None):
        """
        Implementation of Bayesian linear regression. See chapter 3.3 of the book
        "Pattern Recognition and Machine Learning" by Bishop for more details.

        Parameters
        ----------
        alpha: float
            Specifies the variance of the prior for the weights w
        beta : float
            Defines the inverse of the noise, i.e. beta = 1 / sigma^2
        basis_func : function
            Function handle to transfer the input with via basis functions
            (see the code above for an example)
        prior: Prior object
            Prior for alpha and beta. If set to None the default prior is used
        do_mcmc: bool
            If set to true different values for alpha and beta are sampled via MCMC from the marginal log likelihood
            Otherwise the marginal log likelihood is optimized with scipy fmin function
        n_hypers : int
            Number of samples for alpha and beta
        chain_length : int
            The chain length of the MCMC sampler
        burnin_steps: int
            The number of burnin steps before the sampling procedure starts
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.X = None
        self.y = None
        self.alpha = alpha
        self.beta = beta
        self.basis_func = basis_func
        if prior is None:
            self.prior = Prior(rng=self.rng)
        else:
            self.prior = prior
        self.do_mcmc = do_mcmc
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        self.models = None

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

        # Theta is on a log scale
        alpha = np.exp(theta[0])
        beta = 1 / np.exp(theta[1])

        D = self.X_transformed.shape[1]
        N = self.X_transformed.shape[0]

        A = beta * np.dot(self.X_transformed.T, self.X_transformed)
        A += np.eye(self.X_transformed.shape[1]) * alpha
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.linalg.LinAlgError:
             A_inv = np.linalg.inv(A + np.random.rand(A.shape[0], A.shape[1]) * 1e-8)
 

        m = beta * np.dot(A_inv, self.X_transformed.T)
        m = np.dot(m, self.y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.X_transformed, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(A))

        if self.prior is not None:
            mll += self.prior.lnprob(theta)

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
        return -self.marginal_log_likelihood(theta)

    @BaseModel._check_shapes_train
    def train(self, X, y, do_optimize=True):
        """
        First optimized the hyperparameters if do_optimize is True and then computes
        the posterior distribution of the weights. See chapter 3.3 of the book by Bishop
        for more details.

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

        self.X = X

        if self.basis_func is not None:
            self.X_transformed = self.basis_func(X)
        else:
            self.X_transformed = self.X

        self.y = y

        if do_optimize:
            if self.do_mcmc:
                sampler = emcee.EnsembleSampler(self.n_hypers, 2,
                                                self.marginal_log_likelihood)

                # Do a burn-in in the first iteration
                if not self.burned:
                    # Initialize the walkers by sampling from the prior
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)

                    # Run MCMC sampling
                    self.p0, _, _ = sampler.run_mcmc(self.p0,
                                                     self.burnin_steps,
                                                     rstate0=self.rng)

                    self.burned = True

                # Start sampling
                pos, _, _ = sampler.run_mcmc(self.p0,
                                             self.chain_length,
                                             rstate0=self.rng)

                # Save the current position, it will be the start point in
                # the next iteration
                self.p0 = pos

                # Take the last samples from each walker
                self.hypers = np.exp(sampler.chain[:, -1])
            else:
                # Optimize hyperparameters of the Bayesian linear regression        
                res = optimize.fmin(self.negative_mll, self.rng.rand(2))
                self.hypers = [[np.exp(res[0]), np.exp(res[1])]]

        else:
            self.hypers = [[self.alpha, self.beta]]

        self.models = []
        for sample in self.hypers:
            alpha = sample[0]
            beta = sample[1]

            logger.debug("Alpha=%f ; Beta=%f" % (alpha, beta))

            S_inv = beta * np.dot(self.X_transformed.T, self.X_transformed)
            S_inv += np.eye(self.X_transformed.shape[1]) * alpha
            try:
                S = np.linalg.inv(S_inv)
            except np.linalg.linalg.LinAlgError:
                S = np.linalg.inv(S_inv + np.random.rand(S_inv.shape[0], S_inv.shape[1]) * 1e-8)

            m = beta * np.dot(np.dot(S, self.X_transformed.T), self.y)

            self.models.append((m, S))

    @BaseModel._check_shapes_predict
    def predict(self, X_test):
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
        if self.basis_func is not None:
            X_transformed = self.basis_func(X_test)
        else:
            X_transformed = X_test

        # Marginalise predictions over hyperparameters
        mu = np.zeros([len(self.hypers), X_transformed.shape[0]])
        var = np.zeros([len(self.hypers), X_transformed.shape[0]])

        for i, h in enumerate(self.hypers):
            mu[i] = np.dot(self.models[i][0].T, X_transformed.T)
            var[i] = 1. / h[1] + np.diag(np.dot(np.dot(X_transformed, self.models[i][1]), X_transformed.T))

        m = mu.mean(axis=0)
        v = var.mean(axis=0)
        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        return m, v
