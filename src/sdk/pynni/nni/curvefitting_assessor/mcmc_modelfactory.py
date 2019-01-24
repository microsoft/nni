import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]) 

import logging
import numpy as np
import emcee
import inspect
import traceback
from functools import reduce
from scipy.misc import logsumexp
from scipy.stats import norm, kde
from scipy.optimize import curve_fit, leastsq, fmin_bfgs, fmin_l_bfgs_b, nnls
from curvefunctions import curve_combination_models, model_defaults
from ml_curvemodel import recency_weights

logger = logging.getLogger('curvefitting_Assessor: modelfactory')

class MCMCCurveModelCombination(object):
    def __init__(self, ml_curve_models, xlim, burn_in=500, nwalkers=100, nsamples=2500):
        self.ml_curve_models = ml_curve_models
        self.xlim = xlim
        self.burn_in = burn_in
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.rand_init_ball = 1e-6
        self.normalize_weights = True
        self.monotonicity_constraint = True
        self.soft_monotonicity_constraint = False
        self.initial_model_weight_ml_estimate = False
        self.normalized_weights_initialization = 'const'
        self.strictly_positive_weights = True
        self.sanity_check_prior = True
        self.recency_weighting = True
        
    def fit(self, x, y, model_weights=None):
        if self.fit_ml_individual(x, y, model_weights):
            #run MCMC:
            self.fit_mcmc(x, y)
            return True
        else:
            logger.warning('fit_ml_individual failed')
            return False

    def y_lim_sanity_check(self, ylim):
        # just make sure that the prediction is not below 0 nor insanely big
        # HOWEVER: there might be cases where some models might predict value larger than 1.0
        # and this is alright, because in those cases we don't necessarily want to stop a run.
        if not np.isfinite(ylim) or ylim < 0. or ylim > 100.0:
            return False
        else:
            return True

    def fit_ml_individual(self, x, y, model_weights):
        '''
            Do a ML fit for each model individually and then another ML fit for the combination of models.
        '''
        self.fit_models = []
        for model in self.ml_curve_models:
            if model.fit(x, y):
                ylim = model.predict(self.xlim)
                if not self.y_lim_sanity_check(ylim):
                    logger.warning('ML fit of model %s is out of bound range [0.0, 100.] at xlim.', model.function.__name__)
                    continue
                params, sigma = model.split_theta_to_array(model.ml_params)
                if not np.isfinite(self.ln_model_prior(model, params)):
                    logger.warning('ML fit of model %s is not supported by prior.', model.function.__name__)
                    continue
                self.fit_models.append(model)
        if len(self.fit_models) == 0:
            return False

        if model_weights is None:
            if self.normalize_weights:
                if self.normalized_weights_initialization == 'constant':
                    #initialize with a constant value
                    #we will sample in this unnormalized space and then later normalize
                    model_weights = [10. for model in self.fit_models]
                else:# self.normalized_weights_initialization == 'normalized'
                    model_weights = [1./len(self.fit_models) for model in self.fit_models]
            else:
                if self.initial_model_weight_ml_estimate:
                    model_weights = self.get_ml_model_weights(x, y)
                    logger.warning(model_weights)
                    non_zero_fit_models = []
                    non_zero_weights = []
                    for w, model in zip(model_weights, self.fit_models):
                        if w > 1e-4:
                            non_zero_fit_models.append(model)
                            non_zero_weights.append(w)
                    self.fit_models = non_zero_fit_models
                    model_weights = non_zero_weights
                else:
                    model_weights = [1./len(self.fit_models) for model in self.fit_models]

        #build joint ml estimated parameter vector
        model_params = []
        all_model_params = []
        for model in self.fit_models:
            params, sigma = model.split_theta_to_array(model.ml_params)
            model_params.append(params)
            all_model_params.extend(params)
        y_predicted = self.predict_given_params(x, model_params, model_weights)
        sigma = (y - y_predicted).std()

        self.ml_params = self.join_theta(all_model_params, sigma, model_weights)
        self.ndim = len(self.ml_params)
        if self.nwalkers < 2*self.ndim:
            self.nwalkers = 2*self.ndim
            logger.warning('warning: increasing number of walkers to 2*ndim=%d', self.nwalkers)
        return True

    def get_ml_model_weights(self, x, y_target):
        '''
            Get the ML estimate of the model weights.

            Take all the models that have been fit using ML.
            For each model we get a prediction of y: y_i
            Now how can we combine those to reduce the squared error:
                argmin_w (y_target - w_1 * y_1 - w_2 * y_2 - w_3 * y_3 ...)
            Deriving and setting to zero we get a linear system of equations that we need to solve.

            Resource on QP:
            http://stats.stackexchange.com/questions/21565/how-do-i-fit-a-constrained-regression-in-r-so-that-coefficients-total-1
            http://maggotroot.blogspot.de/2013/11/constrained-linear-least-squares-in.html
        '''
        num_models = len(self.fit_models)
        y_predicted = []
        b = []
        for model in self.fit_models:
            y_model = model.predict(x)
            y_predicted.append(y_model)
            b.append(y_model.dot(y_target))
        a = np.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(num_models):
                a[i, j] = y_predicted[i].dot(y_predicted[j])
                #if i == j:
                #    a[i, j] -= 0.1 #constraint the weights!
        a_rank = np.linalg.matrix_rank(a)
        if a_rank != num_models:
            logger.info('Rank %d not sufficcient for solving the linear system. %d needed at least.', a_rank, num_models)
        try:
            logger.info(np.linalg.lstsq(a, b)[0])
            logger.info(np.linalg.solve(a, b))
            ##return np.linalg.solve(a, b)
            logger.info(nnls(a, b)[0])
            #weights = [w if w > 1e-4 else 1e-4 for w in weights]
            weights = nnls(a, b)[0]
            return weights
        #except LinAlgError as e:
        except:
            return [1./len(self.fit_models) for model in self.fit_models]

    #priors
    def ln_prior(self, theta):
        ln = 0
        model_params, sigma, model_weights = self.split_theta(theta)
        for model, params in zip(self.fit_models, model_params):
            ln += self.ln_model_prior(model, params)
        #if self.normalize_weights:
            #when we normalize we expect all weights to be positive
        #we expect all weights to be positive
        if self.strictly_positive_weights and np.any(model_weights < 0):
            return -np.inf
        return ln

    def ln_model_prior(self, model, params):
        if not model.are_params_in_bounds(params):
            return -np.inf
        if self.monotonicity_constraint:
            #check for monotonicity(this obviously this is a hack, but it works for now):
            x_mon = np.linspace(2, self.xlim, 100)
            y_mon = model.function(x_mon, *params)
            if np.any(np.diff(y_mon) < 0):
                return -np.inf
        elif self.soft_monotonicity_constraint:
            #soft monotonicity: defined as the last value being bigger than the first one
            x_mon = np.asarray([2, self.xlim])
            y_mon = model.function(x_mon, *params)
            if y_mon[0] > y_mon[-1]:
                return -np.inf
        ylim = model.function(self.xlim, *params)
        #sanity check for ylim
        if self.sanity_check_prior and not self.y_lim_sanity_check(ylim):
            return -np.inf
        else:
            return 0.0

    #likelihood
    def ln_likelihood(self, theta, x, y):
        y_model, sigma = self.predict_given_theta(x, theta)
        if self.recency_weighting:
            weight = recency_weights(len(y))
            ln_likelihood = (weight*norm.logpdf(y-y_model, loc=0, scale=sigma)).sum()
        else:
            ln_likelihood = norm.logpdf(y-y_model, loc=0, scale=sigma).sum()

        if np.isnan(ln_likelihood):
            return -np.inf
        else:
            return ln_likelihood

    def ln_prob(self, theta, x, y):
        '''
            posterior probability
        '''
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, x, y)

    def split_theta(self, theta):
        '''
            theta is structured as follows:
            for each model i
                for each model parameter j
            theta = (theta_ij, sigma, w_i)
        '''
        num_models = len(self.fit_models)
        model_weights = theta[-len(self.fit_models):]
        all_model_params = []
        for model in self.fit_models:
            num_model_params = len(model.function_params)
            model_params = theta[:num_model_params]
            all_model_params.append(model_params)
            theta = theta[num_model_params:]
        sigma = theta[0]
        model_weights = theta[1:]
        assert len(model_weights) == len(self.fit_models)
        return all_model_params, sigma, model_weights

    def join_theta(self, model_params, sigma, model_weights):
        #assert len(model_params) == len(model_weights)
        theta = []
        theta.extend(model_params)
        theta.append(sigma)
        theta.extend(model_weights)
        return theta

    def fit_mcmc(self, x, y):
        #initialize in an area around the starting position
        assert self.ml_params is not None
        pos = [self.ml_params + self.rand_init_ball*np.random.randn(self.ndim) for i in range(self.nwalkers)]
        sampler = emcee.EnsembleSampler(self.nwalkers,
                self.ndim,
                self.ln_prob,
                args=(x, y))
        sampler.run_mcmc(pos, self.nsamples)
        self.mcmc_chain = sampler.chain
        if self.normalize_weights:
            self.normalize_chain_model_weights()

    def normalize_chain_model_weights(self):
        '''
            In the chain we sample w_1,... w_i however we are interested in the model
            probabilities p_1,... p_i
        '''
        model_weights_chain = self.mcmc_chain[:,:,-len(self.fit_models):]
        model_probabilities_chain = model_weights_chain / model_weights_chain.sum(axis=2)[:,:,np.newaxis]
        #replace in chain
        self.mcmc_chain[:,:,-len(self.fit_models):] = model_probabilities_chain

    def get_burned_in_samples(self):
        samples = self.mcmc_chain[:, self.burn_in:, :].reshape((-1, self.ndim))
        return samples

    def print_probs(self):
        burned_in_chain = self.get_burned_in_samples()
        model_probabilities = burned_in_chain[:,-len(self.fit_models):]
        logger.info(model_probabilities.mean(axis=0))

    def predict_given_theta(self, x, theta):
        '''
            returns y_predicted, sigma
        '''
        model_params, sigma, model_weights = self.split_theta(theta)
        y_predicted = self.predict_given_params(x, model_params, model_weights)
        return y_predicted, sigma

    def predict_given_params(self, x, model_params, model_weights):
        '''
            returns y_predicted
        '''
        if self.normalize_weights:
            model_weight_sum = np.sum(model_weights)
            model_ws = [weight/model_weight_sum for weight in model_weights]
        else:
            model_ws = model_weights

        y_model = []
        for model, model_w, params in zip(self.fit_models, model_ws, model_params):
            y_model.append(model_w*model.function(x, *params))
        y_predicted = reduce(lambda a, b: a+b, y_model)
        return y_predicted

    def prob_x_greater_than(self, x, y, theta):
        '''
            P(f(x) > y | Data, theta)
        '''
        model_params, sigma, model_weights = self.split_theta(theta)

        y_predicted = self.predict_given_params(x, model_params, model_weights)

        cdf = norm.cdf(y, loc=y_predicted, scale=sigma)
        return 1. - cdf

    def posterior_prob_x_greater_than(self, x, y, thin=1):
        '''
            P(f(x) > y | Data)
            thin: only use every thin'th sample
            Posterior probability that f(x) is greater than y.
        '''
        assert isinstance(x, float) or isinstance(x, int)
        assert isinstance(y, float) or isinstance(y, int)
        probs = []
        samples = self.get_burned_in_samples()
        for theta in samples[::thin]:   
            probs.append(self.prob_x_greater_than(x, y, theta))
        return np.ma.masked_invalid(probs).mean()

    def predictive_distribution(self, x, thin=1):
        assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        predictions = []
        for theta in samples[::thin]:
            model_params, sigma, model_weights = self.split_theta(theta)
            y_predicted = self.predict_given_params(x, model_params, model_weights)
            predictions.append(y_predicted)
        return np.asarray(predictions)

    def predictive_ln_prob_distribution(self, x, y, thin=1):
        '''
            posterior log p(y|x,D) for each sample
        '''
        #assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        ln_probs = []
        for theta in samples[::thin]:
            ln_prob = self.ln_likelihood(theta, x, y)
            ln_probs.append(ln_prob)
        return np.asarray(ln_probs)

    def posterior_ln_prob(self, x, y, thin=10):
        '''
            posterior log p(y|x,D)

            1/S sum p(y|D,theta_s)
            equivalent to:
            logsumexp(log p(y|D,theta_s)) - log(S)
        '''
        assert not np.isscalar(x)
        assert not np.isscalar(y)
        x = np.asarray(x)
        y = np.asarray(y)
        ln_probs = self.predictive_ln_prob_distribution(x, y)
        return logsumexp(ln_probs) - np.log(len(ln_probs))

    def predict(self, x, thin=1):
        '''
            E[f(x)]
        '''
        predictions = self.predictive_distribution(x, thin)        
        return np.ma.masked_invalid(predictions).mean()

    def predictive_std(self, x, thin=1):
        '''
           sqrt(Var[f(x)])
        '''
        predictions = self.predictive_distribution(x, thin)
        return np.ma.masked_invalid(predictions).std()

    def serialize(self, fname):
        import pickle
        pickle.dump(self, open(fname, 'wb'))
