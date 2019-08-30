from __future__ import print_function
import argparse

import tensorflow as tf

import numpy as np
import math
import os
import shutil
import time
from matplotlib import pyplot as plt

from nni.compressors.tf_compressor._nnimc_tf import TfPruner
from nni.compressors.tf_compressor._nnimc_tf import _tf_detect_prunable_layers

def add_sparse_args(parser):
    parser.add_argument('--growth', type=str, default='momentum', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude', help='Prune mode / pruning mode. Choose from: magnitude, SET.')
    parser.add_argument('--redistribution', type=str, default='momentum', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / prune rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--dense', action='store_true', help='Enable dense mode. Default: False.')
    parser.add_argument('--verbose', action='store_true', help='Prints verbose status of pruning/growth algorithms.')

class CosineDecay(object):
    """Decays a pruning rate according to a cosine schedule

    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """
    def __init__(self, prune_rate, global_step, eta_min=0.005):
        self.global_step = global_step
        self.decay_steps = 0
        self.learning_rate = eta_min
        self.eta_min = eta_min

    def step(self):
        self.decay_steps += 1

    def get_dr(self, prune_rate):
        lr_decayed = tf.train.cosine_decay(self.learning_rate, self.global_step, self.decay_steps)
        return lr_decayed*prune_rate

class LinearDecay(object):
    """Anneals the pruning rate linearly with each step."""
    def __init__(self, prune_rate, T_max):
        self.steps = 0
        self.decrement = prune_rate/float(T_max)
        self.current_prune_rate = prune_rate

    def step(self):
        self.steps += 1
        self.current_prune_rate -= self.decrement

    def get_dr(self, prune_rate):
        return self.current_prune_rate



class NaiveSparsePruner(TfPruner):
    def __init__(self, optimizer, prune_rate_decay, density=0.05, sparse_init='constant', prune_rate=0.5, prune_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', verbose=False, fp16=False):
        super().__init__()
        growth_modes = ['random', 'momentum', 'momentum_neuron']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.growth_mode = growth_mode
        self.prune_mode = prune_mode
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.verbose = verbose

        self.growth_func = growth_mode
        self.prune_func = prune_mode
        self.redistribution_func = redistribution_mode

        self.global_growth = False
        self.global_prune = False

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.name2removed = {}

        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.prune_rate = prune_rate
        self.name2prune_rate = {}
        self.steps = 0
        self.start_name = None

        # global growth/prune state
        self.prune_threshold = 0.001
        self.growth_threshold = 0.001
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02
        self.prune_every_k_steps = None
        self.half = fp16
        self.name_to_32bit = {}

        self.density = density
        self.sparse_init = sparse_init

        self.sess = tf.Session()
        self.prunable_layers = []
        self.weight_list = {}
        self.weight_shape = {}
        self.assign_weight_handler = []

    def calc_mask(self, layer_info, weight):
        weight_mask = self.masks.get(layer_info.name, None)
        if weight_mask is None:
            return tf.get_variable(layer_info.name+'_mask',initializer=tf.ones(weight.shape), trainable=False)

        return weight_mask
    
    '''
    def init_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True
    '''

    def init(self, mode='constant', density=0.05):
        self.sparsity = density
        
        if mode == 'constant':
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            self.baseline_nonzero = 0
            for layer_info in self.prunable_layers:
                weight_op = layer_info.layer.inputs[layer_info.weight_index].op
                weight = weight_op.inputs[0]
                threshold = tf.contrib.distributions.percentile(tf.abs(weight), density * 100)
                mask = tf.cast(tf.math.greater(tf.abs(weight), threshold), weight.dtype)
                self.masks[layer_info.name] = mask
                self.weight_list[layer_info.name] = weight
                
                shape = weight.get_shape()
                variable_parameters = 1
                for dim in shape:
                    # print(dim)
                    variable_parameters *= dim.value
                self.weight_shape[layer_info.name] = variable_parameters
                self.baseline_nonzero += variable_parameters*density

        total_size = 0
        for layer_info in self.prunable_layers:
            total_size += self.weight_shape[layer_info.name]
        print('Total parameters after removed layers:', total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(density, density*total_size))

    def at_end_of_epoch(self):
        self.truncate_weights()
        
        self.reset_momentum()

    def step(self, sess):
        #self.optimizer.step()
        #self.apply_mask()
        self.sess = sess
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)

        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.truncate_weights()
                self.reset_momentum()
                
    def apply_mask(self):
        for layer_info in self.prunable_layers:
            mask = self.masks[layer_info.name]
            weight = self.weight_list[layer_info.name]
            self.sess.run(tf.assign(weight, mask*weight))

    def bind_model(self, module):
        self.modules.append(module)
        self.prunable_layers = _tf_detect_prunable_layers(module)
        self.init(mode=self.sparse_init, density=self.density)

    def is_at_start_of_pruning(self, name):
        if self.start_name is None: self.start_name = name
        if name == self.start_name: return True
        else: return False

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)

    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                if verbose:
                    print('Removing {0}...'.format(name))
                removed.add(name)
                self.masks.pop(name)
        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed: self.names.pop(i)
            else: i += 1

    def truncate_weights(self):
        self.gather_statistics()
        self.adjust_prune_rate()

        total_nonzero_new = 0
        for layer_info in self.prunable_layers:
            name = layer_info.name
            mask = self.masks[name]

            weight = self.weight_list[name]
            # prune
            new_mask = self.magnitude_prune(mask, weight, name)
            number = self.sess.run(tf.reduce_sum(new_mask))
            removed = self.name2nonzeros[name] - number
            self.total_removed += removed
            self.name2removed[name] = removed
            self.sess.run(tf.assign(self.masks[name], new_mask))
            

        name2regrowth = self.calc_growth_redistribution()
        for layer_info in self.prunable_layers:
            name = layer_info.name
            new_mask = self.masks[name]

            # growth
            new_mask = self.momentum_growth(self, new_mask, layer_info, math.floor(name2regrowth[name]))

            new_nonzero = self.sess.run(tf.reduce_sum(new_mask))

            # exchanging masks
            self.sess.run(tf.assign(self.masks[name], new_mask))
            total_nonzero_new += new_nonzero

        self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        self.adjusted_growth = 0.25*self.adjusted_growth + (0.75*(self.baseline_nonzero - total_nonzero_new)) + np.mean(self.adjustments)
        if self.total_nonzero > 0 and self.verbose:
            print('Nonzero before/after: {0}/{1}. Growth adjustment: {2:.2f}.'.format(
                  self.total_nonzero, total_nonzero_new, self.adjusted_growth))

    def adjust_prune_rate(self):
        for layer_info in self.prunable_layers:
            name = layer_info.name
            if name not in self.name2prune_rate: self.name2prune_rate[name] = self.prune_rate
            self.name2prune_rate[name] = self.prune_rate

            mask_shape = self.weight_shape[name]
            sparsity = self.name2zeros[name]/float(mask_shape)

            if sparsity < 0.2:
                # determine if matrix is relativly dense but still growing
                expected_variance = 1.0/len(list(self.name2variance.keys()))
                actual_variance = self.name2variance[name]
                expected_vs_actual = expected_variance/actual_variance
                if expected_vs_actual < 1.0:
                    # growing
                    self.name2prune_rate[name] = min(sparsity, self.name2prune_rate[name])

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}
        self.name2removed = {}

        self.total_variance = 0.0
        self.total_removed = 0
        self.total_nonzero = 0
        self.total_zero = 0.0
        for layer_info in self.prunable_layers:
            name = layer_info.name
            mask = self.masks[name]
            
            self.name2variance[name] = self.momentum_redistribution(layer_info, mask, self.sess)

            if not np.isnan(self.name2variance[name]):
                self.total_variance += self.name2variance[name]
            
            self.name2nonzeros[name] = self.sess.run(tf.reduce_sum(mask))
            self.name2zeros[name] = self.weight_shape[name] - self.name2nonzeros[name]
            self.total_nonzero += self.name2nonzeros[name]
            self.total_zero += self.name2zeros[name]
        
        for name in self.name2variance:
            if self.total_variance != 0.0:
                self.name2variance[name] /= self.total_variance
            else:
                print('Total variance was zero!')
                print(self.growth_func)
                print(self.prune_func)
                print(self.redistribution_func)
                print(self.name2variance)

    def calc_growth_redistribution(self):
        num_overgrowth = 0
        total_overgrowth = 0
        residual = 0

        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        expected_var = 1.0/len(self.name2variance)
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.name2variance:
                prune_rate = self.name2prune_rate[name]
                num_remove = math.ceil(prune_rate*self.name2nonzeros[name])
                num_nonzero = self.name2nonzeros[name]
                num_zero = self.name2zeros[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(self.name2variance[name]*(self.total_removed+self.adjusted_growth))
                regrowth += mean_residual

                if regrowth > 0.99*max_regrowth:
                    name2regrowth[name] = 0.99*max_regrowth
                    residual += regrowth - name2regrowth[name]
                else:
                    name2regrowth[name] = regrowth
            if len(name2regrowth) == 0: mean_residual = 0
            else:
                mean_residual = residual / len(name2regrowth)
            i += 1

        if i == 1000:
            print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))

        for layer_info in self.prunable_layers:
            if self.prune_mode == 'global_magnitude':
                name = layer_info.name
                expected_removed = self.baseline_nonzero*self.name2prune_rate[name]
                if expected_removed == 0.0:
                    name2regrowth[name] = 0.0
                else:
                    expected_vs_actual = self.total_removed/expected_removed
                    name2regrowth[name] = math.floor(expected_vs_actual*name2regrowth[name])

        return name2regrowth


    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, layer_info):
        temp_name = layer_info.name
        layer_info_scope = temp_name.replace(layer_info.layer.type, '')
        weight = self.weight_list[layer_info.name]
        momentum_list = self.optimizer.variables()
        adam_m1 = 0
        adam_m2 = 0
        for tmp_variables in momentum_list:
            if tmp_variables.name.find(layer_info_scope) >= 0 and tmp_variables.get_shape() == weight.get_shape():
                if tmp_variables.name.endswith('Adam:0'):
                    adam_m1 = self.sess.run(tmp_variables)
                elif tmp_variables.name.endswith('Adam_1:0'):
                    adam_m2 = self.sess.run(tmp_variables)
        
        grad = adam_m1/(tf.sqrt(adam_m2) + 1e-08)
        return grad
    
    def reset_momentum(self):
        momentum_list = self.optimizer.variables()

        for layer_info in self.prunable_layers:
            mask = self.masks[layer_info.name]
            t_mask = 1 - mask

            temp_name = layer_info.name
            layer_info_scope = temp_name.replace(layer_info.layer.type, '')
            weight = self.weight_list[layer_info.name]
            momentum_list = self.optimizer.variables()

            for tmp_variables in momentum_list:
                if tmp_variables.name.find(layer_info_scope) >= 0 and tmp_variables.get_shape() == weight.get_shape():
                    if tmp_variables.name.endswith('Adam:0') or tmp_variables.name.endswith('Adam_1:0'):
                        mean = tf.reduce_mean(tmp_variables*mask)
                        new_momentum = tmp_variables*mask + t_mask*mean
                        self.sess.run(tf.assign(tmp_variables, new_momentum))

    def magnitude_prune(self, mask, weight, name):
        num_remove = math.ceil(self.name2prune_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        
        sparsity = k/self.weight_shape[name]
        threshold = tf.contrib.distributions.percentile(weight.abs(), sparsity * 100)
        new_mask = tf.cast(tf.math.greater(weight.abs(), threshold), weight.dtype)
        return new_mask

        

    def momentum_redistribution(self, layer_info, mask, sess):
        grad = self.get_momentum_for_weight(layer_info)
        mean_magnitude = tf.reduce_mean(tf.abs(grad*mask))
        return sess.run(mean_magnitude)
    
    def momentum_growth(self, mask, layer_info, new_mask, total_regrowth):
        grad = self.get_momentum_for_weight(layer_info)
        new_grad = grad*mask
        k = (total_regrowth/self.weight_shape[layer_info.name])*100
        if k>= 100:
            k=100

        self.weight_list[layer_info.name]
        thresh = tf.contrib.distributions.percentile(tf.abs(new_grad), k)
        new_temp_mask = tf.cast(tf.math.greater(weight.abs(), threshold), new_grad.dtype)
        new_maks = tf.maximum(mask, new_temp_mask)
        return new_mask