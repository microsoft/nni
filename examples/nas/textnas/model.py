# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn as nn
from nni.nas.pytorch import mutables

from ops import ConvBN, LinearCombine, AvgPool, MaxPool, RNN, Attention, BatchNorm
from utils import GlobalMaxPool, GlobalAvgPool


class Layer(mutables.MutableScope):
    def __init__(self, key, prev_keys, hidden_units, choose_from_k, cnn_keep_prob, lstm_keep_prob, att_keep_prob, att_mask):
        super(Layer, self).__init__(key)

        def conv_shortcut(kernel_size):
            return ConvBN(kernel_size, hidden_units, hidden_units, cnn_keep_prob, False, True)

        self.n_candidates = len(prev_keys)
        if self.n_candidates:
            self.prec = mutables.InputChoice(choose_from=prev_keys[-choose_from_k:], n_chosen=1)
        else:
            # first layer, skip input choice
            self.prec = None
        self.op = mutables.LayerChoice([
            conv_shortcut(1),
            conv_shortcut(3),
            conv_shortcut(5),
            conv_shortcut(7),
            AvgPool(3, False, True),
            MaxPool(3, False, True),
            RNN(hidden_units, lstm_keep_prob),
            Attention(hidden_units, 4, att_keep_prob, att_mask)
        ])
        if self.n_candidates:
            self.skipconnect = mutables.InputChoice(choose_from=prev_keys)
        else:
            self.skipconnect = None
        self.bn = BatchNorm(hidden_units, False, True)

    def forward(self, last_layer, prev_layers, mask):
        # pass an extra last_layer to deal with layer 0 (prev_layers is empty)
        if self.prec is None:
            prec = last_layer
        else:
            prec = self.prec(prev_layers[-self.prec.n_candidates:])  # skip first
        out = self.op(prec, mask)
        if self.skipconnect is not None:
            connection = self.skipconnect(prev_layers[-self.skipconnect.n_candidates:])
            if connection is not None:
                out += connection
        out = self.bn(out, mask)
        return out


class Model(nn.Module):
    def __init__(self, embedding, hidden_units=256, num_layers=24, num_classes=5, choose_from_k=5,
                 lstm_keep_prob=0.5, cnn_keep_prob=0.5, att_keep_prob=0.5, att_mask=True,
                 embed_keep_prob=0.5, final_output_keep_prob=1.0, global_pool="avg"):
        super(Model, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.init_conv = ConvBN(1, self.embedding.embedding_dim, hidden_units, cnn_keep_prob, False, True)

        self.layers = nn.ModuleList()
        candidate_keys_pool = []
        for layer_id in range(self.num_layers):
            k = "layer_{}".format(layer_id)
            self.layers.append(Layer(k, candidate_keys_pool, hidden_units, choose_from_k,
                                     cnn_keep_prob, lstm_keep_prob, att_keep_prob, att_mask))
            candidate_keys_pool.append(k)

        self.linear_combine = LinearCombine(self.num_layers)
        self.linear_out = nn.Linear(self.hidden_units, self.num_classes)

        self.embed_dropout = nn.Dropout(p=1 - embed_keep_prob)
        self.output_dropout = nn.Dropout(p=1 - final_output_keep_prob)

        assert global_pool in ["max", "avg"]
        if global_pool == "max":
            self.global_pool = GlobalMaxPool()
        elif global_pool == "avg":
            self.global_pool = GlobalAvgPool()

    def forward(self, inputs):
        sent_ids, mask = inputs
        seq = self.embedding(sent_ids.long())
        seq = self.embed_dropout(seq)

        seq = torch.transpose(seq, 1, 2)  # from (N, L, C) -> (N, C, L)

        x = self.init_conv(seq, mask)
        prev_layers = []

        for layer in self.layers:
            x = layer(x, prev_layers, mask)
            prev_layers.append(x)

        x = self.linear_combine(torch.stack(prev_layers))
        x = self.global_pool(x, mask)
        x = self.output_dropout(x)
        x = self.linear_out(x)
        return x
