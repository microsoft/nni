# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tree-structured Parzen Estimator (TPE) tuner for hyper-parameter optimization.

Paper: https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf
Official code: https://github.com/hyperopt/hyperopt/blob/master/hyperopt/tpe.py

This is a slightly modified re-implementation of the algorithm.
"""

__all__ = ['TpeTuner', 'TpeArguments', 'suggest', 'suggest_parameter']

from collections import defaultdict
import math
from typing import NamedTuple, Optional, Union

import numpy as np
from scipy.special import erf

from nni.tuner import Tuner
from nni.common.hpo_utils import OptimizeMode, format_search_space, deformat_parameters
from . import random_tuner

## Public API part ##

class TpeArguments(NamedTuple):
    """
    These are the hyper-parameters of TPE algorithm itself.
    To avoid confusing with trials' hyper-parameters, they are called "arguments" in this code.

    Parameters
    ==========
    constant_liar_type: 'best' | 'worst' | 'mean' | None (default: 'best')
        TPE algorithm itself does not support parallel tuning.
        This parameter specifies how to optimize for trial_concurrency > 1.

        None (or "null" in YAML) means do not optimize. This is the default behavior in legacy version.

        How each liar works is explained in paper's section 6.1.
        In general "best" suit for small trial number and "worst" suit for large trial number.

    n_startup_jobs: int (default: 20)
        The first N hyper-parameters are generated fully randomly for warming up.
        If the search space is large, you can increase this value.
        Or if max_trial_number is small, you man want to decrease it.

    n_ei_candidates: int (default: 24)
        For each iteration TPE samples EI for N sets of parameters and choose the best one. (loosely speaking)

    linear_forgetting: int (default: 25)
        TPE will lower the weights of old trials.
        This controls how many iterations it takes for a trial to start decay.

    prior_weight: float (default: 1.0)
        TPE uses history trials plus the search space itself (called "prior") to generate parameters.
        For prior weight 1.0, the search space is treated as one good trial.

    gamma: float (default: 0.25)
        Controls how many trials are considered "good".
        The number is calculated as "gamma * sqrt(N)".

    eps: float (default: 1e-12)
        Used to limit minimal σ of internal normal distributions.
    """
    constant_liar_type: Optional[str] = 'best'
    n_startup_jobs: int = 20
    n_ei_candidates: int = 24
    linear_forgetting: int = 25
    prior_weight: float = 1.0
    gamma: float = 0.25
    eps: float = 1e-12

class TpeTuner(Tuner):
    def __init__(self, optimize_mode='minimize', seed=None, tpe_args=None):
        """
        Parameters
        ==========
        optimze_mode: 'minimize' | 'maximize' (default: 'minimize')
            Whether optimize to minimize or maximize trial result.
        seed: int | None
            The random seed.
        tpe_args: dict[string, Any] | None
            Advanced users can use this to customize TPE tuner.
            See `TpeArguments` for details.
        """
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.args = TpeArguments(**(tpe_args or {}))
        self.space = None
        self.liar = create_liar(self.args.constant_liar_type)
        self.rng = np.random.default_rng(seed)

        self._params = {}
        self._running_params = {}
        self._history = defaultdict(list)

    def update_search_space(self, space):
        self.space = format_search_space(space)

    def generate_parameters(self, parameter_id, **kwargs):
        if self.liar and self._running_params:
            history = {key: records.copy() for key, records in self._history.items()}  # copy history to append lies
            lie = self.liar.lie()
            for param in self._running_params.values():
                for key, value in param.items():
                    history[key].append(Record(value, lie))
        else:
            history = self._history

        params = suggest(self.args, self.rng, self.space, history)

        self._params[parameter_id] = params
        self._running_params[parameter_id] = params
        return deformat_parameters(params, self.space)

    def receive_trial_result(self, parameter_id, _parameters, loss, **kwargs):
        if self.optimize_mode is OptimizeMode.Maximize:
            loss = -loss
        if self.liar:
            self.liar.update(loss)
        params = self._running_params.pop(parameter_id)
        for key, value in params.items():
            self._history[key].append(Record(value, loss))

    def trial_end(self, parameter_id, _success, **kwargs):
        self._running_params.pop(parameter_id, None)

def suggest(args, rng, space, history):
    params = {}
    for spec in space.values():
        if spec.is_activated(params):  # nested search space is chosen
            params[spec.key] = suggest_parameter(args, rng, spec, history[spec.key])
    return params

def suggest_parameter(args, rng, spec, parameter_history):
    if len(parameter_history) < args.n_startup_jobs:  # not enough history, still warming up
        return random_tuner.suggest_parameter(rng, spec)

    if spec.categorical:
        return suggest_categorical(args, rng, parameter_history, spec.size)

    if spec.normal_distributed:
        mu = spec.mu
        sigma = spec.sigma
        clip = None
    else:
        mu = (spec.low + spec.high) * 0.5
        sigma = spec.high - spec.low
        clip = (spec.low, spec.high)

    return suggest_numerical(args, rng, parameter_history, mu, sigma, clip)

## Public API part end ##

## Utilities part ##

class Record(NamedTuple):
    param: Union[int, float]
    loss: float

class BestLiar:
    def __init__(self):
        self._best = math.inf

    def update(self, loss):
        self._best = min(self._best, loss)

    def lie(self):
        return self._best

class WorstLiar:
    def __init__(self):
        self._worst = -math.inf

    def update(self, loss):
        self._worst = max(self._worst, loss)

    def lie(self):
        return self._worst

class MeanLiar:
    def __init__(self):
        self._sum = 0.0
        self._n = 0

    def update(self, loss):
        self._sum += loss
        self._n += 1

    def lie(self):
        if self._n == 0:    # when there is no real result, all of history is the same lie
            return 0.0      # so the value doesn't matter
        return self._sum / self._n

def create_liar(liar_type):
    if liar_type is None or liar_type.lower == 'none':
        return None
    liar_classes = {
        'best': BestLiar,
        'worst': WorstLiar,
        'mean': MeanLiar,
    }
    return liar_classes[liar_type.lower()]()

## Utilities part end ##

## Algorithm part ##

def suggest_categorical(args, rng, param_history, size):
    below, above = split_history(args, param_history)

    weights = linear_forgetting_weights(args, len(below))
    counts = np.bincount(below, weights, size)
    p = (counts + args.prior_weight) / sum(counts + args.prior_weight)
    samples = rng.choice(size, args.n_ei_candidates, p=p)
    below_llik = np.log(p[samples])  # llik means log-likelyhood

    weights = linear_forgetting_weights(args, len(above))
    counts = np.bincount(above, weights, size)
    p = (counts + args.prior_weight) / sum(counts + args.prior_weight)
    above_llik = np.log(p[samples])

    return samples[np.argmax(below_llik - above_llik)]

def suggest_numerical(args, rng, param_history, prior_mu, prior_sigma, clip):
    below, above = split_history(args, param_history)

    weights, mus, sigmas = adaptive_parzen_normal(args, below, prior_mu, prior_sigma)
    samples = gmm1(args, rng, weights, mus, sigmas, clip)
    below_llik = gmm1_lpdf(args, samples, weights, mus, sigmas, clip)  # llik means log-likelyhood

    weights, mus, sigmas = adaptive_parzen_normal(args, above, prior_mu, prior_sigma)
    above_llik = gmm1_lpdf(args, samples, weights, mus, sigmas, clip)

    return samples[np.argmax(below_llik - above_llik)]

def split_history(args, param_history):
    n_below = math.ceil(args.gamma * math.sqrt(len(param_history)))
    n_below = min(n_below, args.linear_forgetting)
    order = sorted(range(len(param_history)), key=(lambda i: param_history[i].loss))  # argsort by loss
    below = [param_history[i].param for i in order[:n_below]]
    above = [param_history[i].param for i in order[n_below:]]
    return np.asarray(below), np.asarray(above)

def linear_forgetting_weights(args, n):
    lf = args.linear_forgetting
    if n < lf:
        return np.ones(n)
    else:
        ramp = np.linspace(1.0 / n, 1.0, n - lf)
        flat = np.ones(lf)
        return np.concatenate([ramp, flat])

def adaptive_parzen_normal(args, history_mus, prior_mu, prior_sigma):
    """
    The "Adaptive Parzen Estimator" described in paper section 4.2, for normal distribution.

    Because TPE internally only supports categorical and normal distributed space (domain),
    this function is used for everything other than "choice".

    Parameters
    ==========
    args: TpeArguments
        Directly used: prior_weight
    history_mus: 1-d array of float
        Parameter values evaluated in history.
        These are the "observations" in paper section 4.2. ("placing density in the vicinity of K observations")
    prior_mu: float
        µ (mean, loc) value provided by user.
    piror_sigma: float
        σ (standard deviation, scale) value provided by user.

    Returns
    =======
    Tuple of three 1-d float arrays: (weight, µ, σ).

    The tuple represents n+1 "vicinity of observations" and each one's weight,
    calculated from "n" history and "1" user provided prior.

    The result is sorted by µ.
    """
    mus = np.append(history_mus, prior_mu)
    order = np.argsort(mus)
    mus = mus[order]
    prior_index = np.searchsorted(mus, prior_mu)

    if len(mus) == 1:
        sigmas = np.asarray([prior_sigma])
    elif len(mus) == 2:
        sigmas = np.asarray([prior_simga * 0.5, prior_sigma * 0.5])
        sigmas[prior_index] = prior_sigma
    else:
        l_delta = mus[1:-1] - mus[:-2]
        r_delta = mus[2:] - mus[1:-1]
        sigmas_mid = np.maximum(l_delta, r_delta)
        sigmas = np.concatenate([[mus[1] - mus[0]], sigmas_mid, [mus[-1] - mus[-2]]])
        sigmas[prior_index] = prior_sigma
    # "magic formula" in official implementation
    n = min(100, len(mus) + 1)
    sigmas = np.clip(sigmas, prior_sigma / n, prior_sigma)

    weights = np.append(linear_forgetting_weights(args, len(mus)), args.prior_weight)
    weights = weights[order]

    return weights / np.sum(weights), mus, sigmas

def gmm1(args, rng, weights, mus, sigmas, clip=None):
    """
    Gaussian Mixture Model 1D.
    """
    ret = np.asarray([])
    while len(ret) < args.n_ei_candidates:
        n = args.n_ei_candidates - len(ret)
        active = np.argmax(rng.multinomial(1, weights, n), axis=1)
        samples = rng.normal(mus[active], sigmas[active])
        if clip:
            samples = samples[(clip[0] <= samples) & (samples < clip[1])]
        ret = np.concatenate([ret, samples])
    return ret

def gmm1_lpdf(args, samples, weights, mus, sigmas, clip=None):
    """
    Gaussian Mixture Model 1D's log probability distribution function.
    """
    if clip:
        normal_cdf_low = erf((clip[0] - mus) / np.maximum(np.sqrt(2) * sigmas, args.eps)) * 0.5 + 0.5
        normal_cdf_high = erf((clip[1] - mus) / np.maximum(np.sqrt(2) * sigmas, args.eps)) * 0.5 + 0.5
        p_accept = np.sum(weights * (normal_cdf_high - normal_cdf_low))
    else:
        p_accept = 1

    # normal lpdf
    dist = samples.reshape(-1, 1) - mus
    mahal = (dist / np.maximum(sigmas, args.eps)) ** 2
    z = np.sqrt(2 * np.pi) * sigmas
    coef = weights / z / p_accept
    normal_lpdf = -0.5 * mahal + np.log(coef)

    # log sum rows
    m = normal_lpdf.max(axis=1)
    e = np.exp(normal_lpdf - m.reshape(-1, 1))
    return np.log(e.sum(axis=1)) + m

## Algorithm part end ##
