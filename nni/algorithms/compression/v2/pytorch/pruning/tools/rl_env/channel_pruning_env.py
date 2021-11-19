# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

from nni.algorithms.compression.v2.pytorch.base import PrunerModuleWrapper
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.algorithms.compression.v2.pytorch.utils import unfold_config_list, get_module_by_name
from sklearn.linear_model import LinearRegression

_logger = logging.getLogger(__name__)

# for pruning
def acc_reward(net, acc, flops):
    return acc * 0.01


def acc_flops_reward(net, acc, flops):
    error = (100 - acc) * 0.01
    return -error * np.log(flops)


def calc_mask(sparsity, wrapper, wrapper_idx=None, preserve_idx=None, preserve_round=1):
        """
        Calculate the mask of given layer.
        Parameters
        ----------
        sparsity: float
            pruning ratio,  preserved weight ratio is `1 - sparsity`
        wrapper: PrunerModuleWrapper
            layer wrapper of this layer
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        dict
            dictionary for storing masks, keys of the dict:
            'weight':  weight mask tensor
            'bias': bias mask tensor (optional)
        """
        assert wrapper.type in ['Conv2d', 'Linear'], 'module type {} is not supported!'.format(wrapper.type)
        weight = wrapper.module.weight.data
        bias = None
        if hasattr(wrapper.module, 'bias') and wrapper.module.bias is not None:
            bias = wrapper.module.bias.data

        if wrapper.weight_mask is None:
            mask_weight = torch.ones(weight.size()).type_as(weight).detach()
        else:
            mask_weight = wrapper.weight_mask.clone()
        if bias is not None:
            if wrapper.bias_mask is None:
                mask_bias = torch.ones(bias.size()).type_as(bias).detach()
            else:
                mask_bias = wrapper.bias_mask.clone()
        else:
            mask_bias = None
        mask = {'weight': mask_weight, 'bias': mask_bias}

        num_total = weight.size(1)
        num_prune = int(num_total * sparsity)
        if preserve_round > 1:
            num_preserve = num_total - num_prune
            num_preserve = int(
                math.ceil(num_preserve * 1. / preserve_round) * preserve_round)
            if num_preserve > num_total:
                num_preserve = num_total
            num_prune = num_total - num_preserve

        if (num_total < 2 or num_prune < 1) and preserve_idx is None:
            return mask

        return get_mask(mask, weight, num_preserve, wrapper, wrapper_idx, preserve_idx)


def least_square_sklearn(X, Y):
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    return reg.coef_

def measure_layer_for_pruning(layer, x):
    def get_layer_type(layer):
        layer_str = str(layer)
        return layer_str[:layer_str.find('(')].strip()

    def get_layer_param(model):
        import operator
        import functools

        return sum([functools.reduce(operator.mul, i.size(), 1) for i in model.parameters()])

    multi_add = 1
    type_name = get_layer_type(layer)

    # ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        layer.flops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        layer.params = get_layer_param(layer)
    # ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        layer.flops = weight_ops + bias_ops
        layer.params = get_layer_param(layer)
    return

def get_mask(base_mask, weight, num_preserve, wrapper, wrapper_idx, preserve_idx):
    w = weight.data.cpu().numpy()
    if wrapper.type == 'Linear':
        w = w[:, :, None, None]

    if preserve_idx is None:
        importance = np.abs(w).sum((0, 2, 3))
        # sum magnitude along C_in, sort descend
        sorted_idx = np.argsort(-importance)
        d_prime = num_preserve
        preserve_idx = sorted_idx[:d_prime]  # to preserve index
    else:
        d_prime = len(preserve_idx)

    assert len(preserve_idx) == d_prime
    mask = np.zeros(w.shape[1], bool)
    mask[preserve_idx] = True

    # reconstruct, X, Y <= [N, C]
    X, Y = wrapper.input_feat, wrapper.output_feat
    masked_X = X[:, mask]
    if w.shape[2] == 1:  # 1x1 conv or fc
        rec_weight = least_square_sklearn(X=masked_X, Y=Y)
        rec_weight = rec_weight.reshape(-1, 1, 1, d_prime)  # (C_out, K_h, K_w, C_in')
        rec_weight = np.transpose(rec_weight, (0, 3, 1, 2))  # (C_out, C_in', K_h, K_w)

        rec_weight_pad = np.zeros_like(w)
        # pylint: disable=all
        rec_weight_pad[:, mask, :, :] = rec_weight
        rec_weight = rec_weight_pad

        if wrapper.type == 'Linear':
            rec_weight = rec_weight.squeeze()
            assert len(rec_weight.shape) == 2

        # now assign
        wrapper.module.weight.data = torch.from_numpy(rec_weight).to(weight.device)

    mask_weight = torch.zeros_like(weight)
    if wrapper.type == 'Linear':
        mask_weight[:, preserve_idx] = 1.
        if base_mask['bias'] is not None and wrapper.module.bias is not None:
            mask_bias = torch.ones_like(wrapper.module.bias)
    else:
        mask_weight[:, preserve_idx, :, :] = 1.
        mask_bias = None

    return {'weight': mask_weight.detach(), 'bias': mask_bias}


class ChannelPruningEnv:
    def __init__(self, model, config_list, dummy_input, args):
        self.env_params = {'model_type': 'mobilenet', 'flops_ratio': 0.5, 'lbound': 0.2, 'rbound': 1., 'reward': 'acc_reward',
                           'n_calibration_batches': 60, 'n_points_per_layer': 10, 'channel_round': 8, 'batch_size': 50}
        for key in args:
            assert key in self.env_params.keys(), "Error! Illegal key: {}".format(key)
            self.env_params[key] = args[key]

        #-- self.pruner = pruner
        self.model = model
        self.temp_config_list = []
        #-- print("self.temp_config_list: {}".format(self.temp_config_list))
        self.config_list = unfold_config_list(model, config_list)
        #-- print("self.config_list: {}".format(self.config_list))
        self.dummy_input = dummy_input
        self.batch_size = self.env_params['batch_size']
        self.preserve_ratio = self.env_params['flops_ratio']

        # options from args
        self.args = args
        self.lbound = self.env_params['lbound']
        self.rbound = self.env_params['rbound']

        self.n_calibration_batches = self.env_params['n_calibration_batches']
        self.n_points_per_layer = self.env_params['n_points_per_layer']
        self.channel_round = self.env_params['channel_round']

        # sanity check
        assert self.preserve_ratio > self.lbound, 'Error! You can not achieve preserve_ratio smaller than lbound!'

        # build indexs
        self._build_index()
        self.n_prunable_layer = len(self.prunable_idx)

        # extract information for preparing
        self._extract_layer_information()

        # build embedding (static part)
        self._build_state_embedding()

        # build reward
        self.reset()  # restore weight
        self.org_model_size = sum(self.wsize_list)
        _logger.info('=> original weight size: %.4f M param', self.org_model_size * 1. / 1e6)
        self.org_flops = sum(self.flops_list)
        _logger.info('=> FLOPs:')
        _logger.info([self.layer_info_dict[idx]['flops']/1e6 for idx in sorted(self.layer_info_dict.keys())])
        _logger.info('=> original FLOPs: %.4f M', self.org_flops * 1. / 1e6)

        self.expected_preserve_computation = self.preserve_ratio * self.org_flops
        self.reward = eval(self.env_params['reward'])

        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None
        self.best_masks = None
        self.cur_reward = -math.inf
        self.pruner_generated_masks = None

        self.org_w_size = sum(self.wsize_list)
        
    def set_mask(self, pruner_generated_masks: Dict[str, Dict[str, Tensor]]):
        self.pruner_generated_masks = pruner_generated_masks

    def set_reward(self, reward: float):
        self.cur_reward = reward


    def step(self, act):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        if self.visited[self.cur_ind]:
            action = self.strategy_dict[self.prunable_idx[self.cur_ind]][0]
            preserve_idx = self.index_buffer[self.cur_ind]
        else:
            action = self._action_wall(act)  # percentage to preserve
            preserve_idx = None
        # prune and update action
        named_op, action, d_prime, preserve_idx = self.prune_kernel(self.prunable_idx[self.cur_ind], act, preserve_idx)
        self.temp_config_list.append({"op_names": [named_op], "sparsity": float(1. - act)})
        #-- print("self.temp_config_list {}".format(self.temp_config_list))
        if not self.visited[self.cur_ind]:
            for group in self.shared_idx:
                if self.cur_ind in group:  # set the shared ones
                    for g_idx in group:
                        self.strategy_dict[self.prunable_idx[g_idx]][0] = action
                        self.strategy_dict[self.prunable_idx[g_idx - 1]][1] = action
                        self.visited[g_idx] = True
                        self.index_buffer[g_idx] = preserve_idx.copy()

        self.strategy.append(action)  # save action to strategy
        self.d_prime_list.append(d_prime)

        self.strategy_dict[self.prunable_idx[self.cur_ind]][0] = action
        if self.cur_ind > 0:
            self.strategy_dict[self.prunable_idx[self.cur_ind - 1]][1] = action

        # all the actions are made
        if self._is_final_layer():
            assert len(self.strategy) == len(self.prunable_idx)
            current_flops = self._cur_flops()
            compress_ratio = current_flops * 1. / self.org_flops
            reward = self.cur_reward
            info_set = {'compress_ratio': compress_ratio, 'reward': reward, 'strategy': self.strategy.copy(), 'config_list': copy.deepcopy(self.temp_config_list)}

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                '''
                torch.save(self.model, 'best_model.pth')
                torch.save(self.best_masks, 'best_mask.pth')
                best_model = os.path.join(self.env_params['output'], 'best_model.pth')
                best_mask = os.path.join(self.env_params['output'], 'best_mask.pth')
                #-- self.pruner.export_model(model_path=best_model, mask_path=best_mask)
                '''
                _logger.info('New best reward: %.4f, reward: %.4f, compress: %.4f', self.best_reward, reward, compress_ratio)
                _logger.info('New best policy: %s', self.best_strategy)
                _logger.info('New best d primes: %s', self.best_d_prime_list)
            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            self.temp_config_list = []
            return obs, reward, done, info_set
        
        reward = 0
        done = False
        self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  # the index of next layer
        info_set = {'compress_ratio': self._cur_flops() * 1. / self.org_flops, 'reward': reward, 'strategy': self.strategy.copy(), 'config_list': copy.deepcopy(self.temp_config_list)}
        # build next state (in-place modify)
        self.layer_embedding[self.cur_ind][-3] = self._cur_reduced() * 1. / self.org_flops  # reduced
        self.layer_embedding[self.cur_ind][-2] = sum(self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # rest
        self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action
        obs = self.layer_embedding[self.cur_ind, :].copy()

        return obs, reward, done, info_set

    def reset(self):
        # restore env by loading the checkpoint
        #-- self.pruner.reset(self.checkpoint)
        self.cur_ind = 0
        self.strategy = []  # pruning strategy
        self.d_prime_list = []
        self.strategy_dict = copy.deepcopy(self.min_strategy_dict)
        # reset layer embeddings
        self.layer_embedding[:, -1] = 1.
        self.layer_embedding[:, -2] = 0.
        self.layer_embedding[:, -3] = 0.
        obs = self.layer_embedding[0].copy()
        obs[-2] = sum(self.wsize_list[1:]) * 1. / sum(self.wsize_list)
        # for share index
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}
        return obs

    def prune_kernel(self, op_idx, preserve_ratio, preserve_idx=None):
        # m_list = list(self.model.modules())
        named_list, m_list = [], []
        for name, module in self.model.named_modules():
            named_list.append(name)
            m_list.append(module)
        op = m_list[op_idx]
        named_op = named_list[op_idx]
        assert (0. < preserve_ratio <= 1.)
        # assert type(op) == PrunerModuleWrapper
        '''
        print(self.config_list)
        print(op)
        print(named_op)
        print(op_idx)
        '''
        legal = False
        for config in self.config_list:
            if named_op in config['op_names']:
                legal = not legal
        assert legal, "layer not found."

        if preserve_ratio == 1 or (not self.pruner_generated_masks):  # do not prune
            if (preserve_idx is None) or (len(preserve_idx) == op.weight.size(1)):
                return named_op, 1., op.weight.size(1), None  # should be a full index
        op.input_feat = self.layer_info_dict[op_idx]['input_feat']
        op.output_feat = self.layer_info_dict[op_idx]['output_feat']

        #-- masks = calc_mask(sparsity=1-preserve_ratio, wrapper=op, preserve_idx=preserve_idx, preserve_round=self.self.env_params['channel_round'])
        #-- print("============================")
        #-- print(self.pruner_generated_masks)
        masks = self.pruner_generated_masks[named_op]
        m = masks['weight'].cpu().data
        if type(op) == nn.Conv2d:
            d_prime = (m.sum((0, 2, 3)) > 0).sum().item()
            preserve_idx = np.nonzero((m.sum((0, 2, 3)) > 0).numpy())[0]
        else:
            assert type(op) == nn.Linear
            d_prime = (m.sum(1) > 0).sum().item()
            preserve_idx = np.nonzero((m.sum(1) > 0).numpy())[0]

        op.weight_mask = masks['weight']
        if hasattr(op, 'bias') and op.bias is not None and 'bias' in masks:
            op.bias_mask = masks['bias']

        action = (m == 1).sum().item() / m.numel()
        return named_op, action, d_prime, preserve_idx

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_idx) - 1

    def _action_wall(self, action):
        """
        Limit the action generated by DDPG for this layer by two constraints:
        1. The total flops must meet the flops reduce target.
           For example: the original flops of entire model is 1000, target flops ratio is 0.5, target flops
           is 1000*0.5 = 500. The reduced flops of other layers is 400, so the remaining flops quota is 500-400=100,
           if the total original flops of this layer is 250, then the maximum ratio is 100/250 = 0.4. So the
           action of this layer can not be greater than 0.4.
        2. The action must be greater than lbound which is stored in self.strategy_dict.
        """
        assert len(self.strategy) == self.cur_ind

        action = float(action)
        action = np.clip(action, 0, 1)

        other_comp = 0
        this_comp = 0
        for i, idx in enumerate(self.prunable_idx):
            flop = self.layer_info_dict[idx]['flops']
            buffer_flop = self._get_buffer_flops(idx)

            if i == self.cur_ind - 1:  # TODO: add other member in the set
                this_comp += flop * self.strategy_dict[idx][0]
                # add buffer (but not influenced by ratio)
                other_comp += buffer_flop * self.strategy_dict[idx][0]
            elif i == self.cur_ind:
                this_comp += flop * self.strategy_dict[idx][1]
                # also add buffer here (influenced by ratio)
                this_comp += buffer_flop
            else:
                other_comp += flop * self.strategy_dict[idx][0] * self.strategy_dict[idx][1]
                # add buffer
                other_comp += buffer_flop * self.strategy_dict[idx][0]  # only consider input reduction

        self.expected_min_preserve = other_comp + this_comp * action
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][0])  # impossible (should be)

        return action

    def _get_buffer_flops(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
        return buffer_flop

    def _cur_flops(self):
        flops = 0
        for idx in self.prunable_idx:
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            flops += self.layer_info_dict[idx]['flops'] * c * n
            # add buffer computation
            flops += self._get_buffer_flops(idx) * c  # only related to input channel reduction
        return flops

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    def _build_index(self):
        """
        Build following information/data for later pruning:
        self.prunable_idx: layer indices for pruable layers, the index values are the index
            of list(self.model.modules()). Pruable layers are pointwise Conv2d layers and Linear
            layers.
        self.prunable_ops: prunable modules
        self.buffer_idx: layer indices for buffer layers which refers the depthwise layers.
            Each depthwise layer is always followd by a pointwise layer for both mobilenet and
            mobilenetv2. The depthwise layer's filters are pruned when its next pointwise layer's
            corresponding input channels are pruned.
        self.shared_idx: layer indices for layers which share input.
            For example: [[1,4], [8, 10, 15]] means layer 1 and 4 share same input, and layer
            8, 10 and 15 share another input.
        self.org_channels: number of input channels for each layer
        self.min_strategy_dict: key is layer index, value is a tuple, the first value is the minimum
            action of input channel, the second value is the minimum action value of output channel.
        self.strategy_dict: same as self.min_strategy_dict, but it will be updated later.
        """
        self.prunable_idx = []
        self.prunable_ops = []
        self.layer_type_dict = {}
        self.strategy_dict = {}
        self.buffer_dict = {}
        this_buffer_list = []
        self.org_channels = []
        # build index and the min strategy dict
        for i, m in enumerate(self.model.modules()):
            # if isinstance(m, PrunerModuleWrapper):
            for config in self.config_list:
                for name in config['op_names']:
                    #if isinstance(m, type(get_module_by_name(self.model, name)[-1])):
                    if m == get_module_by_name(self.model, name)[-1]:
                        if type(m) == nn.Conv2d and m.groups == m.in_channels:  # depth-wise conv, buffer
                            this_buffer_list.append(i)
                        else:  # really prunable
                            self.prunable_idx.append(i)
                            self.prunable_ops.append(m)
                            self.layer_type_dict[i] = type(m)
                            self.buffer_dict[i] = this_buffer_list
                            this_buffer_list = []  # empty
                            self.org_channels.append(m.in_channels if type(m) == nn.Conv2d else m.in_features)
                            self.strategy_dict[i] = [self.lbound, self.lbound]
        self.strategy_dict[self.prunable_idx[0]][0] = 1  # modify the input
        self.strategy_dict[self.prunable_idx[-1]][1] = 1  # modify the output

        self.shared_idx = []
        if self.env_params['model_type'] == 'mobilenetv2':  # TODO: to be tested! Share index for residual connection
            connected_idx = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]  # to be partitioned
            last_ch = -1
            share_group = None
            for c_idx in connected_idx:
                if self.prunable_ops[c_idx].in_channels != last_ch:  # new group
                    last_ch = self.prunable_ops[c_idx].in_channels
                    if share_group is not None:
                        self.shared_idx.append(share_group)
                    share_group = [c_idx]
                else:  # same group
                    share_group.append(c_idx)
            self.shared_idx.append(share_group)
            _logger.info('=> Conv layers to share channels: %s', self.shared_idx)

        self.min_strategy_dict = copy.deepcopy(self.strategy_dict)

        self.buffer_idx = []
        for _, v in self.buffer_dict.items():
            self.buffer_idx += v

        _logger.info('=> Prunable layer idx: %s', self.prunable_idx)
        _logger.info('=> Buffer layer idx: %s', self.buffer_idx)
        _logger.info('=> Shared idx: %s', self.shared_idx)
        _logger.info('=> Initial min strategy dict: %s', self.min_strategy_dict)

        # added for supporting residual connections during pruning
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}

    def _extract_layer_information(self):
        m_list = list(self.model.modules())

        self.data_saver = []
        self.layer_info_dict = dict()
        self.wsize_list = []
        self.flops_list = []

        # from .lib.utils import measure_layer_for_pruning

        # extend the forward fn to record layer info
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y
            return lambda_forward

        device = None
        for idx in self.prunable_idx + self.buffer_idx:  # get all
            m = m_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)
            # if device is None and type(m) == PrunerModuleWrapper:
            for config in self.config_list:
                for name in config['op_names']:
                    if m == get_module_by_name(self.model, name)[-1]:
                        device = m.weight.device

        # now let the image flow
        _logger.info('=> Extracting information...')
        with torch.no_grad():
            '''
            for i_b, (inputs, target) in enumerate(self._val_loader):  # use image from train set
                if i_b == self.n_calibration_batches:
                    break
                # self.data_saver.append((inputs.clone(), target.clone()))
                input_var = torch.autograd.Variable(inputs).to(device)
            '''
            for i_b in range(self.n_calibration_batches):
                # torch.randn(2, 3, 224, 224) for imagenet else torch.randn(2, 3, 32, 32)
                input_var = torch.autograd.Variable(self.dummy_input).to(device)

                # inference and collect stats
                _ = self.model(input_var)

                if i_b == 0:  # first batch
                    for idx in self.prunable_idx + self.buffer_idx:
                        self.layer_info_dict[idx] = dict()
                        self.layer_info_dict[idx]['params'] = m_list[idx].params
                        self.layer_info_dict[idx]['flops'] = m_list[idx].flops
                        self.wsize_list.append(m_list[idx].params)
                        self.flops_list.append(m_list[idx].flops)
                    _logger.info('flops: %s', self.flops_list)
                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    f_out_np = m_list[idx].output_feat.data.cpu().numpy()
                    # TODO More layer types need support
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save, f_out2save = None, None
                        elif m_list[idx].weight.size(3) > 1:  # normal conv
                            f_in2save, f_out2save = f_in_np, f_out_np
                        else:  # 1x1 conv
                            # assert f_out_np.shape[2] == f_in_np.shape[2]  # now support k=3
                            randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
                            randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer)
                            # input: [N, C, H, W]
                            self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
                            self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer - 100, -1)

                            f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer - 100, -1)
                    else:
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                        f_out2save = f_out_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                        self.layer_info_dict[idx]['output_feat'] = f_out2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))
                        self.layer_info_dict[idx]['output_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['output_feat'], f_out2save))

    def _build_state_embedding(self):
        # build the static part of the state embedding
        _logger.info('Building state embedding...')
        layer_embedding = []
        module_list = list(self.model.modules())
        for i, ind in enumerate(self.prunable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d:
                this_state.append(i)  # index
                this_state.append(0)  # layer type, 0 for conv
                this_state.append(m.in_channels)  # in channels
                this_state.append(m.out_channels)  # out channels
                this_state.append(m.stride[0])  # stride
                this_state.append(m.kernel_size[0])  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size
            elif type(m) == nn.Linear:
                this_state.append(i)  # index
                this_state.append(1)  # layer type, 1 for fc
                this_state.append(m.in_features)  # in channels
                this_state.append(m.out_features)  # out channels
                this_state.append(0)  # stride
                this_state.append(1)  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size

            # this 3 features need to be changed later
            this_state.append(0.)  # reduced
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}
            layer_embedding.append(np.array(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        _logger.info('=> shape of embedding (n_layer * n_dim): %s', layer_embedding.shape)
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding
