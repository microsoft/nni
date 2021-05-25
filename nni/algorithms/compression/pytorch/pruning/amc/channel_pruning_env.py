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

from nni.compression.pytorch.compressor import PrunerModuleWrapper
from .. import AMCWeightMasker

_logger = logging.getLogger(__name__)

# for pruning
def acc_reward(net, acc, flops):
    return acc * 0.01


def acc_flops_reward(net, acc, flops):
    error = (100 - acc) * 0.01
    return -error * np.log(flops)


class ChannelPruningEnv:
    """
    Env for channel pruning search.
    This class is used to prune model using specified pruner. It prunes one layer when
    step() is called. When the last layer is pruned, it evaluate the pruned model using
    evaluator, and use the returned value of evaluator as reward of the episode.

    Usage:
        env = ChannelPruningEnv(pruner, evaluator, val_loader, checkpoint, env_args)
        episode = 0
        T = []
        while episode < num_episode:
            action = agent.select_action(observation)
            observation2, reward, done, info = env.step(action)
            T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

            if done: # end of episode, last layer pruned
                episode += 1
                # train agent with episode data
                for _, s_t, s_t1, a_t, done in T:
                    agent.observe(final_reward, s_t, s_t1, a_t, done)
                    agent.update_policy()
                T = []

    Attributes:
        prunable_idx: layer indices for pruable layers, the index values are the index
            of list(self.model.modules()). Pruable layers are pointwise Conv2d layers and Linear
            layers.
        buffer_idx: layer indices for buffer layers which refers the depthwise layers.
            Each depthwise layer is always followd by a pointwise layer for both mobilenet and
            mobilenetv2. The depthwise layer's filters are pruned when its next pointwise layer's
            corresponding input channels are pruned.
        shared_idx: layer indices for layers which share input.
            For example: [[1,4], [8, 10, 15]] means layer 1 and 4 share same input, and layer
            8, 10 and 15 share another input.
        layer_embedding: embeddings for each prunable layers, the embedding is used as
            observation for DDPG agent.
        layer_info_dict: flops and number of parameters of each layer.
        min_strategy_dict: key is layer index, value is a tuple, the first value is the minimum
            action of input channel, the second value is the minimum action value of output channel.
        strategy_dict: key is layer index, value is a tuple, the first value is the action of input
            channel, the second value is the action of output channel.

    Parameters:
        pruner: Pruner
            NNI Pruner instance used to prune model.
        evaluator: function
            function to evaluate the pruned model.
            The prototype of the function:
                >>> def evaluator(val_loader, model):
                >>>     ...
                >>>     return acc
        val_loader: torch.utils.data.DataLoader
            Data loader of validation dataset.
        checkpoint: dict
            checkpoint of the model to be pruned. It is used to reset model at beginning of each
            episode.
        args:
            A Namespace object containing following arguments:
            model_type: str
                model type to prune, currently 'mobilenet', 'mobilenetv2' and 'resnet' are supported.
            flops_ratio: float
                preserve flops ratio.
            lbound: float
                minimum weight preserve ratio for each layer.
            rbound: float
                maximum weight preserve ratio for each layer.
            reward: function
                reward function type

            # parameters for channel pruning
            n_calibration_batches: int
                number of batches to extract layer information.
            n_points_per_layer: int
                number of feature points per layer.
            channel_round: int
                round channel to multiple of channel_round.

    """
    def __init__(self, pruner, evaluator, val_loader, checkpoint, args):
        self.pruner = pruner
        self.model = pruner.bound_model
        self.checkpoint = checkpoint
        self.batch_size = val_loader.batch_size
        self.preserve_ratio = args.preserve_ratio
        self.channel_prune_masker = AMCWeightMasker(self.model, self.pruner, args.channel_round)

        # options from args
        self.args = args
        self.lbound = args.lbound
        self.rbound = args.rbound

        self.n_calibration_batches = args.n_calibration_batches
        self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round

        # sanity check
        assert self.preserve_ratio > self.lbound, 'Error! You can not achieve preserve_ratio smaller than lbound!'

        # prepare data
        self._val_loader = val_loader
        self._validate = evaluator

        # build indexs
        self._build_index()
        self.n_prunable_layer = len(self.prunable_idx)

        # extract information for preparing
        self._extract_layer_information()

        # build embedding (static part)
        self._build_state_embedding()

        # build reward
        self.reset()  # restore weight
        self.org_acc = self._validate(self._val_loader, self.model)
        _logger.info('=> original acc: %.3f', self.org_acc)
        self.org_model_size = sum(self.wsize_list)
        _logger.info('=> original weight size: %.4f M param', self.org_model_size * 1. / 1e6)
        self.org_flops = sum(self.flops_list)
        _logger.info('=> FLOPs:')
        _logger.info([self.layer_info_dict[idx]['flops']/1e6 for idx in sorted(self.layer_info_dict.keys())])
        _logger.info('=> original FLOPs: %.4f M', self.org_flops * 1. / 1e6)

        self.expected_preserve_computation = self.preserve_ratio * self.org_flops

        self.reward = eval(args.reward)

        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None
        self.best_masks = None

        self.org_w_size = sum(self.wsize_list)

    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        if self.visited[self.cur_ind]:
            action = self.strategy_dict[self.prunable_idx[self.cur_ind]][0]
            preserve_idx = self.index_buffer[self.cur_ind]
        else:
            action = self._action_wall(action)  # percentage to preserve
            preserve_idx = None
        # prune and update action
        action, d_prime, preserve_idx = self.prune_kernel(self.prunable_idx[self.cur_ind], action, preserve_idx)
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
            acc_t1 = time.time()
            acc = self._validate(self._val_loader, self.model)
            acc_t2 = time.time()
            self.val_time = acc_t2 - acc_t1
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'accuracy': acc, 'strategy': self.strategy.copy()}
            reward = self.reward(self, acc, current_flops)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                best_model = os.path.join(self.args.output, 'best_model.pth')
                best_mask = os.path.join(self.args.output, 'best_mask.pth')
                self.pruner.export_model(model_path=best_model, mask_path=best_mask)
                _logger.info('New best reward: %.4f, acc: %.4f, compress: %.4f', self.best_reward, acc, compress_ratio)
                _logger.info('New best policy: %s', self.best_strategy)
                _logger.info('New best d primes: %s', self.best_d_prime_list)
            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            return obs, reward, done, info_set

        info_set = None
        reward = 0
        done = False
        self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  # the index of next layer
        # build next state (in-place modify)
        self.layer_embedding[self.cur_ind][-3] = self._cur_reduced() * 1. / self.org_flops  # reduced
        self.layer_embedding[self.cur_ind][-2] = sum(self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # rest
        self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action
        obs = self.layer_embedding[self.cur_ind, :].copy()

        return obs, reward, done, info_set

    def reset(self):
        # restore env by loading the checkpoint
        self.pruner.reset(self.checkpoint)
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
        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0
        # for share index
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}
        return obs

    def prune_kernel(self, op_idx, preserve_ratio, preserve_idx=None):
        m_list = list(self.model.modules())
        op = m_list[op_idx]
        assert (0. < preserve_ratio <= 1.)
        assert type(op) == PrunerModuleWrapper
        if preserve_ratio == 1:  # do not prune
            if (preserve_idx is None) or (len(preserve_idx) == op.module.weight.size(1)):
                return 1., op.module.weight.size(1), None  # should be a full index
        op.input_feat = self.layer_info_dict[op_idx]['input_feat']
        op.output_feat = self.layer_info_dict[op_idx]['output_feat']

        masks = self.channel_prune_masker.calc_mask(sparsity=1-preserve_ratio, wrapper=op, preserve_idx=preserve_idx)
        m = masks['weight_mask'].cpu().data
        if type(op.module) == nn.Conv2d:
            d_prime = (m.sum((0, 2, 3)) > 0).sum().item()
            preserve_idx = np.nonzero((m.sum((0, 2, 3)) > 0).numpy())[0]
        else:
            assert type(op.module) == nn.Linear
            d_prime = (m.sum(1) > 0).sum().item()
            preserve_idx = np.nonzero((m.sum(1) > 0).numpy())[0]

        op.weight_mask = masks['weight_mask']
        if hasattr(op.module, 'bias') and op.module.bias is not None and 'bias_mask' in masks:
            op.bias_mask = masks['bias_mask']

        action = (m == 1).sum().item() / m.numel()
        return action, d_prime, preserve_idx

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
            if isinstance(m, PrunerModuleWrapper):
                m = m.module
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
        if self.args.model_type == 'mobilenetv2':  # TODO: to be tested! Share index for residual connection
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

        from .lib.utils import measure_layer_for_pruning

        # extend the forward fn to record layer info
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                #TODO replace this flops counter with nni.compression.torch.utils.counter.count_flops_params
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
            if device is None and type(m) == PrunerModuleWrapper:
                device = m.module.weight.device

        # now let the image flow
        _logger.info('=> Extracting information...')
        with torch.no_grad():
            for i_b, (inputs, target) in enumerate(self._val_loader):  # use image from train set
                if i_b == self.n_calibration_batches:
                    break
                self.data_saver.append((inputs.clone(), target.clone()))
                input_var = torch.autograd.Variable(inputs).to(device)

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
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save, f_out2save = None, None
                        elif m_list[idx].module.weight.size(3) > 1:  # normal conv
                            f_in2save, f_out2save = f_in_np, f_out_np
                        else:  # 1x1 conv
                            # assert f_out_np.shape[2] == f_in_np.shape[2]  # now support k=3
                            randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
                            randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer)
                            # input: [N, C, H, W]
                            self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
                            self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer, -1)

                            f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
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
            m = module_list[ind].module
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

