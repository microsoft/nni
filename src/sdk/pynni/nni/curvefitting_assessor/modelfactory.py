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
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from curvefunctions import *

NUM_OF_FUNCTIONS = 12
MAXFEV = 1000000
NUM_OF_SIMULATION_TIME = 20
NUM_OF_INSTANCE = 10
STEP_SIZE = 0.0005


class CurveModel(object):
    def __init__(self, target_pos=20):
        self.target_pos = target_pos

    def fit_theta(self):
        '''use least squares to fit all default curves parameter seperately'''
        x = range(1, self.point_num + 1)
        y = self.trial_history
        for i in range (NUM_OF_FUNCTIONS):
            model = curve_combination_models[i]
            try:
                if model_para_num[model] == 2:
                    a, b = optimize.curve_fit(all_models[model], x, y, maxfev = MAXFEV)[0]
                    model_para[model][0] = a
                    model_para[model][1] = b
                elif model_para_num[model] == 3:
                    a, b, c = optimize.curve_fit(all_models[model], x, y, maxfev = MAXFEV)[0]
                    model_para[model][0] = a
                    model_para[model][1] = b
                    model_para[model][2] = c
                elif model_para_num[model] == 4:
                    a, b, c, d = optimize.curve_fit(all_models[model], x, y, maxfev = MAXFEV)[0]
                    model_para[model][0] = a
                    model_para[model][1] = b
                    model_para[model][2] = c
                    model_para[model][3] = d
            except Exception as exception:
                pass
    
    def filter_curve(self):
        '''filter the poor performing curve'''
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
            if y < median + 3 * std and y > median - 3 * std:
                self.effective_model.append(model)

    def predict_y(self, model, pos):
        '''return the predict y of 'model' when epoch = pos'''
        if model_para_num[model] == 2:
            y = all_models[model](pos, model_para[model][0], model_para[model][1])
        elif model_para_num[model] == 3:
            y = all_models[model](pos, model_para[model][0], model_para[model][1], model_para[model][2])
        elif model_para_num[model] == 4:
            y = all_models[model](pos, model_para[model][0], model_para[model][1], model_para[model][2], model_para[model][3])
        return y
        
    def f_comb(self, pos, sample):
        '''return the value of the f_comb when epoch = pos'''
        ret = 0
        for i in range (self.effective_model_num):
            model = self.effective_model[i]
            y = self.predict_y(model, pos)
            ret += sample[i] * y
        return ret

    def normalize_weight(self, samples):
        '''normalize weight '''
        for i in range (NUM_OF_INSTANCE):
            sum = 0
            for j in range (self.effective_model_num):
                sum += samples[i][j]
            for j in range (self.effective_model_num):
                samples[i][j] /= sum
        return samples

    def sigma_sq(self, sample):
        '''returns the value of sigma square, given the weight's sample'''
        ret = 0
        for i in range(1, self.point_num + 1):
            temp = self.trial_history[i - 1] - self.f_comb(i, sample)
            ret += temp * temp
        return 1.0 * ret / self.point_num

    def normal_distribution(self, pos, sample):
        '''returns the value of normal distribution, given the weight's sample and target position'''
        curr_sigma_sq = self.sigma_sq(sample)
        return np.exp(np.square(self.trial_history[pos - 1] - self.f_comb(pos, sample)) / (-2.0 * curr_sigma_sq)) / np.sqrt(2 * np.pi * np.sqrt(curr_sigma_sq))

    def likelihood(self, samples):
        '''likelihood'''
        ret = np.ones(NUM_OF_INSTANCE)
        for i in range (NUM_OF_INSTANCE):
            for j in range(1, self.point_num + 1):
                ret[i] *= self.normal_distribution(j, samples[i])
        return ret

    def prior(self, samples):
        '''priori distribution'''
        ret = np.ones(NUM_OF_INSTANCE)
        for i in range (NUM_OF_INSTANCE):
            for j in range (self.effective_model_num):
                if not samples[i][j] > 0:
                    ret[i] = 0
            if not self.f_comb(1, samples[i]) < self.f_comb(self.target_pos, samples[i]):
                ret[i] = 0
        return ret

    def target_distribution(self, samples):
        '''posterior probability'''
        curr_likelihood = self.likelihood(samples)
        curr_prior = self.prior(samples)
        ret = np.ones(NUM_OF_INSTANCE)
        for i in range (NUM_OF_INSTANCE):
            ret[i] = curr_likelihood[i] * curr_prior[i]
        return ret

    def MCMC_sampling(self):
        '''
        Adjust the weight of each function using mcmc sampling.
        The initial value of each weight is evenly distribute.
        Reference paper: Tobias Domhan, Jost Tobias Springenberg, Frank Hutter. Speeding up Automatic Hyperparameter Optimization of 
        Deep Neural Networks by Extrapolation of Learning Curves. IJCAI, 2015.
        '''
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
            for i in range (NUM_OF_INSTANCE):
                new_values[i] = self.weight_samples[i] * (1 - change_value_flag[i]) + new_values[i] * change_value_flag[i]
            self.weight_samples = new_values

    def predict(self, trial_history):
        '''predict the value of target position'''
        self.trial_history = trial_history
        self.point_num = len(trial_history)
        self.effective_model = []
        self.fit_theta()
        self.filter_curve()
        self.effective_model_num = len(self.effective_model)
        if self.effective_model_num < 4:
            '''different curve's predictions are too scattered, requires more information'''
            return -1
        self.MCMC_sampling()
        ret = 0
        for i in range(NUM_OF_INSTANCE):
            ret += self.f_comb(self.target_pos, self.weight_samples[i])
        return ret / NUM_OF_INSTANCE