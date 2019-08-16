from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import math
import os
import shutil
import time
from matplotlib import pyplot as plt
from sparselearning.funcs import redistribution_funcs, growth_funcs, prune_funcs

from nni.compressors.torchCompressor._nnimc_torch import TorchPruner

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
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, prune_rate):
        return self.sgd.param_groups[0]['lr']

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



class SparsePruner(TorchPruner):
    def __init__(self, optimizer, prune_rate_decay, prune_rate=0.5, prune_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', verbose=False, fp16=False):
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

    def calc_mask(self, layer_info, weight):
        weight_mask = self.masks.get(layer_info.name, None)
        if weight_mask is None:
            return torch.ones(weight.shape)
        
        if self.half:
            return weight_mask.half()

        return weight_mask
        
    def init_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True

    def init(self, mode='constant', density=0.05):
        self.sparsity = density
        self.init_growth_prune_and_redist()
        self.init_optimizer()
        if mode == 'constant':
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()
                    self.baseline_nonzero += weight.numel()*density
            #self.apply_mask()
        elif mode == 'resume':
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print((weight != 0.0).sum().item())
                    if name in self.name_to_32bit:
                        print('W2')
                    self.masks[name][:] = (weight != 0.0).float().data.cuda()
                    self.baseline_nonzero += weight.numel()*density
            #self.apply_mask()
        elif mode == 'linear':
            # initialization used in sparse evolutionary training
            # scales the number of non-zero weights linearly proportional
            # to the product of all dimensions, that is input*output
            # for fully connected layers, and h*w*in_c*out_c for conv
            # layers.

            total_params = 0
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    total_params += weight.numel()
                    self.baseline_nonzero += weight.numel()*density

            target_params = total_params *density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not ((current_params+tolerance > target_params) and (current_params-tolerance < target_params)):
                new_nonzeros = 0.0
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth =  epsilon*sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95

            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                growth =  epsilon*sum(weight.shape)
                prob = growth/np.prod(weight.shape)
                self.masks[name][:] = (torch.rand(weight.shape) < prob).float().data.cuda()
            #self.apply_mask()

        self.print_nonzero_counts()

        total_size = 0
        for name, module in self.modules[0].named_modules():
            if hasattr(module, 'weight'):
                total_size += module.weight.numel()
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    total_size += module.bias.numel()
        print('Total Model parameters:', total_size)

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total parameters after removed layers:', total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(density, density*total_size))

    def init_growth_prune_and_redist(self):
        if isinstance(self.growth_func, str) and self.growth_func in growth_funcs:
            if 'global' in self.growth_func: self.global_growth = True
            self.growth_func = growth_funcs[self.growth_func]
        elif isinstance(self.growth_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Growth mode function not known: {0}.'.format(self.growth_func))
            print('Use either a custom growth function or one of the pre-defined functions:')
            for key in growth_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown growth mode.')

        if isinstance(self.prune_func, str) and self.prune_func in prune_funcs:
            if 'global' in self.prune_func: self.global_prune = True
            self.prune_func = prune_funcs[self.prune_func]
        elif isinstance(self.prune_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Prune mode function not known: {0}.'.format(self.prune_func))
            print('Use either a custom prune function or one of the pre-defined functions:')
            for key in prune_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown prune mode.')

        if isinstance(self.redistribution_func, str) and self.redistribution_func in redistribution_funcs:
            self.redistribution_func = redistribution_funcs[self.redistribution_func]
        elif isinstance(self.redistribution_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Redistribution mode function not known: {0}.'.format(self.redistribution_func))
            print('Use either a custom redistribution function or one of the pre-defined functions:')
            for key in redistribution_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown redistribution mode.')

    def at_end_of_epoch(self):
        self.truncate_weights()
        if self.verbose:
            self.print_nonzero_counts()
        self.reset_momentum()

    def step(self):
        self.optimizer.step()
        #self.apply_mask()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)

        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.truncate_weights()
                self.reset_momentum()
                if self.verbose:
                    self.print_nonzero_counts()

    def add_module(self, module, density, sparse_init='constant'):
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
        self.remove_weight_partial_name('bias')
        self.remove_type(nn.BatchNorm2d)
        self.remove_type(nn.BatchNorm1d)
        self.remove_type(nn.PReLU)
        self.init(mode=sparse_init, density=density)

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


    def remove_type(self, nn_type, verbose=False):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight_partial_name(name, verbose=True)
    '''
    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    if not self.half:
                        tensor.data = tensor.data*self.masks[name]
                    else:
                        tensor.data = tensor.data*self.masks[name].half()
                        if name in self.name_to_32bit:
                            tensor2 = self.name_to_32bit[name]
                            tensor2.data = tensor2.data*self.masks[name]
    '''

    def adjust_prune_rate(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if name not in self.name2prune_rate: self.name2prune_rate[name] = self.prune_rate

                self.name2prune_rate[name] = self.prune_rate

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                if sparsity < 0.2:
                    # determine if matrix is relativly dense but still growing
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        # growing
                        self.name2prune_rate[name] = min(sparsity, self.name2prune_rate[name])

    def truncate_weights(self):
        self.gather_statistics()
        self.adjust_prune_rate()

        total_nonzero_new = 0
        if self.global_prune:
            self.total_removed = self.prune_func(self)
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # prune
                    new_mask = self.prune_func(self, mask, weight, name)
                    removed = self.name2nonzeros[name] - new_mask.sum().item()
                    self.total_removed += removed
                    self.name2removed[name] = removed
                    self.masks[name][:] = new_mask

        name2regrowth = self.calc_growth_redistribution()
        if self.global_growth:
            total_nonzero_new = self.growth_func(self, self.total_removed + self.adjusted_growth)
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()

                    # growth
                    new_mask = self.growth_func(self, name, new_mask, math.floor(name2regrowth[name]), weight)
                    new_nonzero = new_mask.sum().item()

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()
                    total_nonzero_new += new_nonzero
        #self.apply_mask()

        # Some growth techniques and redistribution are probablistic and we might not grow enough weights or too much weights
        # Here we run an exponential smoothing over (prune-growth) residuals to adjust future growth
        self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        self.adjusted_growth = 0.25*self.adjusted_growth + (0.75*(self.baseline_nonzero - total_nonzero_new)) + np.mean(self.adjustments)
        if self.total_nonzero > 0 and self.verbose:
            print('Nonzero before/after: {0}/{1}. Growth adjustment: {2:.2f}.'.format(
                  self.total_nonzero, total_nonzero_new, self.adjusted_growth))

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}
        self.name2removed = {}

        self.total_variance = 0.0
        self.total_removed = 0
        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]

                # redistribution
                self.name2variance[name] = self.redistribution_func(self, name, weight, mask)

                if not np.isnan(self.name2variance[name]):
                    self.total_variance += self.name2variance[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
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

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if self.prune_mode == 'global_magnitude':
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
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                if name in self.name2variance:
                    val = '{0}: {1}->{2}, density: {3:.3f}, proportion: {4:.4f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()), self.name2variance[name])
                    print(val)
                else:
                    print(name, num_nonzeros)

        print('Prune rate: {0}\n'.format(self.prune_rate))

    def reset_momentum(self):
        """
        Taken from: https://github.com/AlliedToasters/synapses/blob/master/synapses/SET_layer.py
        Resets buffers from memory according to passed indices.
        When connections are reset, parameters should be treated
        as freshly initialized.
        """
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                weights = list(self.optimizer.state[tensor])
                for w in weights:
                    if w == 'momentum_buffer':
                        # momentum
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])
                    elif w == 'square_avg' or \
                        w == 'exp_avg' or \
                        w == 'exp_avg_sq' or \
                        w == 'exp_inf':
                        # Adam
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])
