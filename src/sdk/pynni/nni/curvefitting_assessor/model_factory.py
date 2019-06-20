# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
import numpy as np
from scipy import optimize
from .curvefunctions import *

# Number of curve functions we prepared, more details can be found in "curvefunctions.py"
NUM_OF_FUNCTIONS = 12
# Number of simulation time when we do MCMC sampling
NUM_OF_SIMULATION_TIME = 20
# Number of samples we select when we do MCMC sampling
NUM_OF_INSTANCE = 10
# The step size of each noise when we do MCMC sampling
STEP_SIZE = 0.0005
# Number of least fitting function, if effective function is lower than this number, we will ask for more information
LEAST_FITTED_FUNCTION = 4

logger = logging.getLogger('curvefitting_Assessor')

class CurveModel(object):
    """Build a Curve Model to predict the performance

    Algorithm: https://github.com/Microsoft/nni/blob/master/src/sdk/pynni/nni/curvefitting_assessor/README.md

    Parameters
    ----------
    target_pos: int
        The point we need to predict
    """
    def __init__(self, target_pos):
        self.target_pos = target_pos
        self.trial_history = []
        self.point_num = 0
        self.effective_model = []
        self.effective_model_num = 0
        self.weight_samples = []

    def fit_theta(self):
        """use least squares to fit all default curves parameter seperately

        Returns
        -------
        None
        """
        x = range(1, self.point_num + 1)
        y = self.trial_history
        for i in range(NUM_OF_FUNCTIONS):
            model = curve_combination_models[i]
            try:
                # The maximum number of iterations to fit is 100*(N+1), where N is the number of elements in `x0`.
                if model_para_num[model] == 2:
                    a, b = optimize.curve_fit(all_models[model], x, y)[0]
                    model_para[model][0] = a
                    model_para[model][1] = b
                elif model_para_num[model] == 3:
                    a, b, c = optimize.curve_fit(all_models[model], x, y)[0]
                    model_para[model][0] = a
                    model_para[model][1] = b
                    model_para[model][2] = c
                elif model_para_num[model] == 4:
                    a, b, c, d = optimize.curve_fit(all_models[model], x, y)[0]
                    model_para[model][0] = a
                    model_para[model][1] = b
                    model_para[model][2] = c
                    model_para[model][3] = d
            except (RuntimeError, FloatingPointError, OverflowError, ZeroDivisionError):
                # Ignore exceptions caused by numerical calculations
                pass
            except Exception as exception:
                logger.critical("Exceptions in fit_theta:", exception)

    def filter_curve(self):
        """filter the poor performing curve

        Returns
        -------
        None
        """
        avg = np.sum(self.trial_history) / self.point_num
        standard = avg * avg * self.point_num
        predict_data = []
        tmp_model = []
        for i in range(NUM_OF_FUNCTIONS):
            var = 0
            model = curve_combination_models[i]
            for j in range(1, self.point_num + 1):
                y = self.predict_y(model, j)
                var += (y - self.trial_history[j - 1]) * (y - self.trial_history[j - 1])
            if var < standard:
                predict_data.append(y)
                tmp_model.append(curve_combination_models[i])
        median = np.median(predict_data)
        std = np.std(predict_data)
        for model in tmp_model:
            y = self.predict_y(model, self.target_pos)
            epsilon = self.point_num / 10 * std
            if y < median + epsilon and y > median - epsilon:
                self.effective_model.append(model)
        self.effective_model_num = len(self.effective_model)
        logger.info('List of effective model: ', self.effective_model)

    def predict_y(self, model, pos):
        """return the predict y of 'model' when epoch = pos

        Parameters
        ----------
        model: string
            name of the curve function model
        pos: int
            the epoch number of the position you want to predict

        Returns
        -------
        int:
            The expected matrix at pos
        """
        if model_para_num[model] == 2:
            y = all_models[model](pos, model_para[model][0], model_para[model][1])
        elif model_para_num[model] == 3:
            y = all_models[model](pos, model_para[model][0], model_para[model][1], model_para[model][2])
        elif model_para_num[model] == 4:
            y = all_models[model](pos, model_para[model][0], model_para[model][1], model_para[model][2], model_para[model][3])
        return y

    def f_comb(self, pos, sample):
        """return the value of the f_comb when epoch = pos

        Parameters
        ----------
        pos: int
            the epoch number of the position you want to predict
        sample: list
            sample is a (1 * NUM_OF_FUNCTIONS) matrix, representing{w1, w2, ... wk}

        Returns
        -------
        int
            The expected matrix at pos with all the active function's prediction
        """
        ret = 0
        for i in range(self.effective_model_num):
            model = self.effective_model[i]
            y = self.predict_y(model, pos)
            ret += sample[i] * y
        return ret

    def normalize_weight(self, samples):
        """normalize weight

        Parameters
        ----------
        samples: list
            a collection of sample, it's a (NUM_OF_INSTANCE * NUM_OF_FUNCTIONS) matrix,
            representing{{w11, w12, ..., w1k}, {w21, w22, ... w2k}, ...{wk1, wk2,..., wkk}}

        Returns
        -------
        list
            samples after normalize weight
        """
        for i in range(NUM_OF_INSTANCE):
            total = 0
            for j in range(self.effective_model_num):
                total += samples[i][j]
            for j in range(self.effective_model_num):
                samples[i][j] /= total
        return samples

    def sigma_sq(self, sample):
        """returns the value of sigma square, given the weight's sample

        Parameters
        ----------
        sample: list
            sample is a (1 * NUM_OF_FUNCTIONS) matrix, representing{w1, w2, ... wk}

        Returns
        -------
        float
            the value of sigma square, given the weight's sample
        """
        ret = 0
        for i in range(1, self.point_num + 1):
            temp = self.trial_history[i - 1] - self.f_comb(i, sample)
            ret += temp * temp
        return 1.0 * ret / self.point_num

    def normal_distribution(self, pos, sample):
        """returns the value of normal distribution, given the weight's sample and target position

        Parameters
        ----------
        pos: int
            the epoch number of the position you want to predict
        sample: list
            sample is a (1 * NUM_OF_FUNCTIONS) matrix, representing{w1, w2, ... wk}

        Returns
        -------
        float
            the value of normal distribution
        """
        curr_sigma_sq = self.sigma_sq(sample)
        delta = self.trial_history[pos - 1] - self.f_comb(pos, sample)
        return np.exp(np.square(delta) / (-2.0 * curr_sigma_sq)) / np.sqrt(2 * np.pi * np.sqrt(curr_sigma_sq))

    def likelihood(self, samples):
        """likelihood

        Parameters
        ----------
        sample: list
            sample is a (1 * NUM_OF_FUNCTIONS) matrix, representing{w1, w2, ... wk}

        Returns
        -------
        float
            likelihood
        """
        ret = np.ones(NUM_OF_INSTANCE)
        for i in range(NUM_OF_INSTANCE):
            for j in range(1, self.point_num + 1):
                ret[i] *= self.normal_distribution(j, samples[i])
        return ret

    def prior(self, samples):
        """priori distribution

        Parameters
        ----------
        samples: list
            a collection of sample, it's a (NUM_OF_INSTANCE * NUM_OF_FUNCTIONS) matrix,
            representing{{w11, w12, ..., w1k}, {w21, w22, ... w2k}, ...{wk1, wk2,..., wkk}}

        Returns
        -------
        float
            priori distribution
        """
        ret = np.ones(NUM_OF_INSTANCE)
        for i in range(NUM_OF_INSTANCE):
            for j in range(self.effective_model_num):
                if not samples[i][j] > 0:
                    ret[i] = 0
            if self.f_comb(1, samples[i]) >= self.f_comb(self.target_pos, samples[i]):
                ret[i] = 0
        return ret

    def target_distribution(self, samples):
        """posterior probability

        Parameters
        ----------
        samples: list
            a collection of sample, it's a (NUM_OF_INSTANCE * NUM_OF_FUNCTIONS) matrix,
            representing{{w11, w12, ..., w1k}, {w21, w22, ... w2k}, ...{wk1, wk2,..., wkk}}

        Returns
        -------
        float
            posterior probability
        """
        curr_likelihood = self.likelihood(samples)
        curr_prior = self.prior(samples)
        ret = np.ones(NUM_OF_INSTANCE)
        for i in range(NUM_OF_INSTANCE):
            ret[i] = curr_likelihood[i] * curr_prior[i]
        return ret

    def mcmc_sampling(self):
        """Adjust the weight of each function using mcmc sampling.
        The initial value of each weight is evenly distribute.
        Brief introduction:
        (1)Definition of sample:
            Sample is a (1 * NUM_OF_FUNCTIONS) matrix, representing{w1, w2, ... wk}
        (2)Definition of samples:
            Samples is a collection of sample, it's a (NUM_OF_INSTANCE * NUM_OF_FUNCTIONS) matrix,
            representing{{w11, w12, ..., w1k}, {w21, w22, ... w2k}, ...{wk1, wk2,..., wkk}}
        (3)Definition of model:
            Model is the function we chose right now. Such as: 'wap', 'weibull'.
        (4)Definition of pos:
            Pos is the position we want to predict, corresponds to the value of epoch.

        Returns
        -------
        None
        """
        init_weight = np.ones((self.effective_model_num), dtype=np.float) / self.effective_model_num
        self.weight_samples = np.broadcast_to(init_weight, (NUM_OF_INSTANCE, self.effective_model_num))
        for i in range(NUM_OF_SIMULATION_TIME):
            # sample new value from Q(i, j)
            new_values = np.random.randn(NUM_OF_INSTANCE, self.effective_model_num) * STEP_SIZE + self.weight_samples
            new_values = self.normalize_weight(new_values)
            # compute alpha(i, j) = min{1, P(j)Q(j, i)/P(i)Q(i, j)}
            alpha = np.minimum(1, self.target_distribution(new_values) / self.target_distribution(self.weight_samples))
            # sample u
            u = np.random.rand(NUM_OF_INSTANCE)
            # new value
            change_value_flag = (u < alpha).astype(np.int)
            for j in range(NUM_OF_INSTANCE):
                new_values[j] = self.weight_samples[j] * (1 - change_value_flag[j]) + new_values[j] * change_value_flag[j]
            self.weight_samples = new_values

    def predict(self, trial_history):
        """predict the value of target position

        Parameters
        ----------
        trial_history: list
            The history performance matrix of each trial.

        Returns
        -------
        float
            expected final result performance of this hyperparameter config
        """
        self.trial_history = trial_history
        self.point_num = len(trial_history)
        self.fit_theta()
        self.filter_curve()
        if self.effective_model_num < LEAST_FITTED_FUNCTION:
            # different curve's predictions are too scattered, requires more information
            return None
        self.mcmc_sampling()
        ret = 0
        for i in range(NUM_OF_INSTANCE):
            ret += self.f_comb(self.target_pos, self.weight_samples[i])
        return ret / NUM_OF_INSTANCE
