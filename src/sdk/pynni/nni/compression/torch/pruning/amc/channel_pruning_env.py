# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from copy import deepcopy
import math
import copy
import numpy as np
import torch
import torch.nn as nn

#from .lib.utils import AverageMeter, accuracy, prGreen
from .lib.utils import prGreen

# for pruning
def acc_reward(net, acc, flops):
    return acc * 0.01


def acc_flops_reward(net, acc, flops):
    error = (100 - acc) * 0.01
    return -error * np.log(flops)


class ChannelPruningEnv:
    """
    Env for channel pruning search
    """
    def __init__(self, model, val_func, val_loader, preserve_ratio, args, export_model=False, use_new_input=False):
        # default setting
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]

        # save options
        self.model = model
        #self.checkpoint = checkpoint
        self.checkpoint = deepcopy(model.state_dict())
        #self.n_data_worker = n_data_worker
        self.batch_size = val_loader.batch_size
        #self.data_type = data
        #print('data:', data)
        self.preserve_ratio = preserve_ratio

        # options from args
        self.args = args
        self.lbound = args.lbound
        self.rbound = args.rbound

        self.use_real_val = args.use_real_val

        self.n_calibration_batches = args.n_calibration_batches
        self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        #self.data_root = args.data_root

        self.export_model = export_model
        self.use_new_input = use_new_input

        # sanity check
        assert self.preserve_ratio > self.lbound, 'Error! You can make achieve preserve_ratio smaller than lbound!'

        # prepare data
        #self._init_data()
        self.val_loader = val_loader
        self._validate = val_func

        # build indexs
        self._build_index()
        self.n_prunable_layer = len(self.prunable_idx)

        # extract information for preparing
        self._extract_layer_information()

        # build embedding (static part)
        self._build_state_embedding()

        # build reward
        self.reset()  # restore weight
        self.org_acc = self._validate(self.val_loader, self.model)
        print('=> original acc: {:.3f}%'.format(self.org_acc))
        self.org_model_size = sum(self.wsize_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_model_size * 1. / 1e6))
        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print([self.layer_info_dict[idx]['flops']/1e6 for idx in sorted(self.layer_info_dict.keys())])
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))

        self.expected_preserve_computation = self.preserve_ratio * self.org_flops

        self.reward = eval(args.reward)

        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None

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

        if self.export_model:  # export checkpoint
            print('# Pruning {}: ratio: {}, d_prime: {}'.format(self.cur_ind, action, d_prime))

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
            acc = self._validate(self.val_loader, self.model)
            acc_t2 = time.time()
            self.val_time = acc_t2 - acc_t1
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'accuracy': acc, 'strategy': self.strategy.copy()}
            reward = self.reward(self, acc, current_flops)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                prGreen('New best reward: {:.4f}, acc: {:.4f}, compress: {:.4f}'.format(self.best_reward, acc, compress_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            if self.export_model:  # export state dict
                torch.save(self.model.state_dict(), self.export_path)
                return None, None, None, None
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
        self.model.load_state_dict(self.checkpoint)
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

    def set_export_path(self, path):
        self.export_path = path

    def prune_kernel(self, op_idx, preserve_ratio, preserve_idx=None):
        '''Return the real ratio'''
        m_list = list(self.model.modules())
        op = m_list[op_idx]
        assert (preserve_ratio <= 1.)

        if preserve_ratio == 1:  # do not prune
            return 1., op.weight.size(1), None  # TODO: should be a full index
            # n, c, h, w = op.weight.size()
            # mask = np.ones([c], dtype=bool)

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        n, c = op.weight.size(0), op.weight.size(1)
        d_prime = format_rank(c * preserve_ratio)
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > c:
            d_prime = int(np.floor(c * 1. / self.channel_round) * self.channel_round)

        extract_t1 = time.time()
        if self.use_new_input:  # this is slow and may lead to overfitting
            self._regenerate_input_feature()
        X = self.layer_info_dict[op_idx]['input_feat']  # input after pruning of previous ops
        Y = self.layer_info_dict[op_idx]['output_feat']  # fixed output from original model
        weight = op.weight.data.cpu().numpy()
        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None]
        extract_t2 = time.time()
        self.extract_time += extract_t2 - extract_t1
        fit_t1 = time.time()

        if preserve_idx is None:  # not provided, generate new
            importance = np.abs(weight).sum((0, 2, 3))
            sorted_idx = np.argsort(-importance)  # sum magnitude along C_in, sort descend
            preserve_idx = sorted_idx[:d_prime]  # to preserve index
        assert len(preserve_idx) == d_prime
        mask = np.zeros(weight.shape[1], bool)
        mask[preserve_idx] = True

        # reconstruct, X, Y <= [N, C]
        masked_X = X[:, mask]
        if weight.shape[2] == 1:  # 1x1 conv or fc
            from .lib.utils import least_square_sklearn
            rec_weight = least_square_sklearn(X=masked_X, Y=Y)
            rec_weight = rec_weight.reshape(-1, 1, 1, d_prime)  # (C_out, K_h, K_w, C_in')
            rec_weight = np.transpose(rec_weight, (0, 3, 1, 2))  # (C_out, C_in', K_h, K_w)
        else:
            raise NotImplementedError('Current code only supports 1x1 conv now!')
        if not self.export_model:  # pad, pseudo compress
            rec_weight_pad = np.zeros_like(weight)
            rec_weight_pad[:, mask, :, :] = rec_weight
            rec_weight = rec_weight_pad

        if op_type == 'Linear':
            rec_weight = rec_weight.squeeze()
            assert len(rec_weight.shape) == 2
        fit_t2 = time.time()
        self.fit_time += fit_t2 - fit_t1
        # now assign
        op.weight.data = torch.from_numpy(rec_weight)#.cuda()
        action = np.sum(mask) * 1. / len(mask)  # calculate the ratio
        if self.export_model:  # prune previous buffer ops
            prev_idx = self.prunable_idx[self.prunable_idx.index(op_idx) - 1]
            for idx in range(prev_idx, op_idx):
                m = m_list[idx]
                if type(m) == nn.Conv2d:  # depthwise
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :, :, :])#.cuda()
                    if m.groups == m.in_channels:
                        m.groups = int(np.sum(mask))
                elif type(m) == nn.BatchNorm2d:
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask])#.cuda()
                    m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask])#.cuda()
                    m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask])#.cuda()
                    m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask])#.cuda()
        return action, d_prime, preserve_idx

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_idx) - 1

    def _action_wall(self, action):
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
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            flops += self.layer_info_dict[idx]['flops'] * c * n
            # add buffer computation
            flops += self._get_buffer_flops(idx) * c  # only related to input channel reduction
        return flops

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    #def _init_data(self):
        # split the train set into train + val
        # for CIFAR, split 5k for val
        # for ImageNet, split 3k for val
        #val_size = 5000 if 'cifar' in self.data_type else 3000
        #self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
        #                                                                self.n_data_worker, val_size,
        #                                                                data_root=self.data_root,
        #                                                                use_real_val=self.use_real_val,
        #                                                                shuffle=False)  # same sampling
        #if self.use_real_val:  # use the real val set for eval, which is actually wrong
        #    print('*** USE REAL VALIDATION SET!')

    def _build_index(self):
        self.prunable_idx = []
        self.prunable_ops = []
        self.layer_type_dict = {}
        self.strategy_dict = {}
        self.buffer_dict = {}
        this_buffer_list = []
        self.org_channels = []
        # build index and the min strategy dict
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.prunable_layer_types:
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
        if self.args.model == 'mobilenetv2':  # TODO: to be tested! Share index for residual connection
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
            print('=> Conv layers to share channels: {}'.format(self.shared_idx))

        self.min_strategy_dict = copy.deepcopy(self.strategy_dict)

        self.buffer_idx = []
        for k, v in self.buffer_dict.items():
            self.buffer_idx += v

        print('=> Prunable layer idx: {}'.format(self.prunable_idx))
        print('=> Buffer layer idx: {}'.format(self.buffer_idx))
        print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))

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
                measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        for idx in self.prunable_idx + self.buffer_idx:  # get all
            m = m_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)

        # now let the image flow
        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.val_loader):  # use image from train set
                if i_b == self.n_calibration_batches:
                    break
                self.data_saver.append((input.clone(), target.clone()))
                input_var = torch.autograd.Variable(input) #.cuda()

                # inference and collect stats
                _ = self.model(input_var)

                if i_b == 0:  # first batch
                    for idx in self.prunable_idx + self.buffer_idx:
                        self.layer_info_dict[idx] = dict()
                        self.layer_info_dict[idx]['params'] = m_list[idx].params
                        self.layer_info_dict[idx]['flops'] = m_list[idx].flops
                        self.wsize_list.append(m_list[idx].params)
                        self.flops_list.append(m_list[idx].flops)
                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    f_out_np = m_list[idx].output_feat.data.cpu().numpy()
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

    def _regenerate_input_feature(self):
        # only re-generate the input feature
        m_list = list(self.model.modules())

        # delete old features
        for k, v in self.layer_info_dict.items():
            if 'input_feat' in v:
                v.pop('input_feat')

        # now let the image flow
        print('=> Regenerate features...')

        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.data_saver):
                input_var = torch.autograd.Variable(input)#.cuda()

                # inference and collect stats
                _ = self.model(input_var)

                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save = None
                        else:
                            randx = self.layer_info_dict[idx][(i_b, 'randx')]
                            randy = self.layer_info_dict[idx][(i_b, 'randy')]
                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:  # fc
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))

    def _build_state_embedding(self):
        # build the static part of the state embedding
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
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

