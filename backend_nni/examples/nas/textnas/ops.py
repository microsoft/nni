# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
from torch import nn

from utils import get_length, INF


class Mask(nn.Module):
    def forward(self, seq, mask):
        # seq: (N, C, L)
        # mask: (N, L)
        seq_mask = torch.unsqueeze(mask, 2)
        seq_mask = torch.transpose(seq_mask.repeat(1, 1, seq.size()[1]), 1, 2)
        return seq.where(torch.eq(seq_mask, 1), torch.zeros_like(seq))


class BatchNorm(nn.Module):
    def __init__(self, num_features, pre_mask, post_mask, eps=1e-5, decay=0.9, affine=True):
        super(BatchNorm, self).__init__()
        self.mask_opt = Mask()
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=1.0 - decay, affine=affine)

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.bn(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq


class ConvBN(nn.Module):
    def __init__(self, kernal_size, in_channels, out_channels, cnn_keep_prob,
                 pre_mask, post_mask, with_bn=True, with_relu=True):
        super(ConvBN, self).__init__()
        self.mask_opt = Mask()
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv1d(in_channels, out_channels, kernal_size, 1, bias=True, padding=(kernal_size - 1) // 2)
        self.dropout = nn.Dropout(p=(1 - cnn_keep_prob))

        if with_bn:
            self.bn = BatchNorm(out_channels, not post_mask, True)

        if with_relu:
            self.relu = nn.ReLU()

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.conv(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        if self.with_bn:
            seq = self.bn(seq, mask)
        if self.with_relu:
            seq = self.relu(seq)
        seq = self.dropout(seq)
        return seq


class AvgPool(nn.Module):
    def __init__(self, kernal_size, pre_mask, post_mask):
        super(AvgPool, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernal_size, 1, padding=(kernal_size - 1) // 2)
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.mask_opt = Mask()

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.avg_pool(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq


class MaxPool(nn.Module):
    def __init__(self, kernal_size, pre_mask, post_mask):
        super(MaxPool, self).__init__()
        self.max_pool = nn.MaxPool1d(kernal_size, 1, padding=(kernal_size - 1) // 2)
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.mask_opt = Mask()

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.max_pool(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq


class Attention(nn.Module):
    def __init__(self, num_units, num_heads, keep_prob, is_mask):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.keep_prob = keep_prob

        self.linear_q = nn.Linear(num_units, num_units)
        self.linear_k = nn.Linear(num_units, num_units)
        self.linear_v = nn.Linear(num_units, num_units)

        self.bn = BatchNorm(num_units, True, is_mask)
        self.dropout = nn.Dropout(p=1 - self.keep_prob)

    def forward(self, seq, mask):
        in_c = seq.size()[1]
        seq = torch.transpose(seq, 1, 2)  # (N, L, C)
        queries = seq
        keys = seq
        num_heads = self.num_heads

        # T_q = T_k = L
        Q = F.relu(self.linear_q(seq))  # (N, T_q, C)
        K = F.relu(self.linear_k(seq))  # (N, T_k, C)
        V = F.relu(self.linear_v(seq))  # (N, T_k, C)

        # Split and concat
        Q_ = torch.cat(torch.split(Q, in_c // num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, in_c // num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, in_c // num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)
        # Key Masking
        key_masks = mask.repeat(num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1)  # (h*N, 1, T_k)
        key_masks = key_masks.repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(outputs) * (-INF)  # extremely small value
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)

        query_masks = mask.repeat(num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, -1)  # (h*N, T_q, 1)
        query_masks = query_masks.repeat(1, 1, keys.size()[1]).float()  # (h*N, T_q, T_k)

        att_scores = F.softmax(outputs, dim=-1) * query_masks  # (h*N, T_q, T_k)
        att_scores = self.dropout(att_scores)

        # Weighted sum
        x_outputs = torch.matmul(att_scores, V_)  # (h*N, T_q, C/h)
        # Restore shape
        x_outputs = torch.cat(
            torch.split(x_outputs, x_outputs.size()[0] // num_heads, dim=0),
            dim=2)  # (N, T_q, C)

        x = torch.transpose(x_outputs, 1, 2)  # (N, C, L)
        x = self.bn(x, mask)

        return x


class RNN(nn.Module):
    def __init__(self, hidden_size, output_keep_prob):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.bid_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_keep_prob = output_keep_prob

        self.out_dropout = nn.Dropout(p=(1 - self.output_keep_prob))

    def forward(self, seq, mask):
        # seq: (N, C, L)
        # mask: (N, L)
        max_len = seq.size()[2]
        length = get_length(mask)
        seq = torch.transpose(seq, 1, 2)  # to (N, L, C)
        packed_seq = nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True,
                                                       enforce_sorted=False)
        outputs, _ = self.bid_rnn(packed_seq)
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True,
                                                   total_length=max_len)[0]
        outputs = outputs.view(-1, max_len, 2, self.hidden_size).sum(2)  # (N, L, C)
        outputs = self.out_dropout(outputs)  # output dropout
        return torch.transpose(outputs, 1, 2)  # back to: (N, C, L)


class LinearCombine(nn.Module):
    def __init__(self, layers_num, trainable=True, input_aware=False, word_level=False):
        super(LinearCombine, self).__init__()
        self.input_aware = input_aware
        self.word_level = word_level

        if input_aware:
            raise NotImplementedError("Input aware is not supported.")
        self.w = nn.Parameter(torch.full((layers_num, 1, 1, 1), 1.0 / layers_num),
                              requires_grad=trainable)

    def forward(self, seq):
        nw = F.softmax(self.w, dim=0)
        seq = torch.mul(seq, nw)
        seq = torch.sum(seq, dim=0)
        return seq
