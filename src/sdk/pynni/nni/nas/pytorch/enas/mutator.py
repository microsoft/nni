# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.mutator import Mutator
from nni.nas.pytorch.mutables import LayerChoice, InputChoice, MutableScope


class StackedLSTMCell(nn.Module):
    def __init__(self, layers, size, bias):
        super().__init__()
        self.lstm_num_layers = layers
        self.lstm_modules = nn.ModuleList([nn.LSTMCell(size, size, bias=bias)
                                           for _ in range(self.lstm_num_layers)])

    def forward(self, inputs, hidden):
        prev_c, prev_h = hidden
        next_c, next_h = [], []
        for i, m in enumerate(self.lstm_modules):
            curr_c, curr_h = m(inputs, (prev_c[i], prev_h[i]))
            next_c.append(curr_c)
            next_h.append(curr_h)
            inputs = curr_h[-1]
        return next_c, next_h


class EnasMutator(Mutator):

    def __init__(self, model, lstm_size=64, lstm_num_layers=1, tanh_constant=1.5, cell_exit_extra_step=False,
                 skip_target=0.4, branch_bias=0.25):
        super().__init__(model)
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.cell_exit_extra_step = cell_exit_extra_step
        self.skip_target = skip_target
        self.branch_bias = branch_bias

        self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)
        self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1)
        self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]), requires_grad=False)  # pylint: disable=not-callable
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.bias_dict = nn.ParameterDict()

        self.max_layer_choice = 0
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                if self.max_layer_choice == 0:
                    self.max_layer_choice = mutable.length
                assert self.max_layer_choice == mutable.length, \
                    "ENAS mutator requires all layer choice have the same number of candidates."
                # We are judging by keys and module types to add biases to layer choices. Needs refactor.
                if "reduce" in mutable.key:
                    def is_conv(choice):
                        return "conv" in str(type(choice)).lower()
                    bias = torch.tensor([self.branch_bias if is_conv(choice) else -self.branch_bias  # pylint: disable=not-callable
                                         for choice in mutable.choices])
                    self.bias_dict[mutable.key] = nn.Parameter(bias, requires_grad=False)

        self.embedding = nn.Embedding(self.max_layer_choice + 1, self.lstm_size)
        self.soft = nn.Linear(self.lstm_size, self.max_layer_choice, bias=False)

    def sample_search(self):
        self._initialize()
        self._sample(self.mutables)
        return self._choices

    def sample_final(self):
        return self.sample_search()

    def _sample(self, tree):
        mutable = tree.mutable
        if isinstance(mutable, LayerChoice) and mutable.key not in self._choices:
            self._choices[mutable.key] = self._sample_layer_choice(mutable)
        elif isinstance(mutable, InputChoice) and mutable.key not in self._choices:
            self._choices[mutable.key] = self._sample_input_choice(mutable)
        for child in tree.children:
            self._sample(child)
        if isinstance(mutable, MutableScope) and mutable.key not in self._anchors_hid:
            if self.cell_exit_extra_step:
                self._lstm_next_step()
            self._mark_anchor(mutable.key)

    def _initialize(self):
        self._choices = dict()
        self._anchors_hid = dict()
        self._inputs = self.g_emb.data
        self._c = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self._h = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self.sample_log_prob = 0
        self.sample_entropy = 0
        self.sample_skip_penalty = 0

    def _lstm_next_step(self):
        self._c, self._h = self.lstm(self._inputs, (self._c, self._h))

    def _mark_anchor(self, key):
        self._anchors_hid[key] = self._h[-1]

    def _sample_layer_choice(self, mutable):
        self._lstm_next_step()
        logit = self.soft(self._h[-1])
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        if mutable.key in self.bias_dict:
            logit += self.bias_dict[mutable.key]
        branch_id = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
        log_prob = self.cross_entropy_loss(logit, branch_id)
        self.sample_log_prob += torch.sum(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += torch.sum(entropy)
        self._inputs = self.embedding(branch_id)
        return F.one_hot(branch_id, num_classes=self.max_layer_choice).bool().view(-1)

    def _sample_input_choice(self, mutable):
        query, anchors = [], []
        for label in mutable.choose_from:
            if label not in self._anchors_hid:
                self._lstm_next_step()
                self._mark_anchor(label)  # empty loop, fill not found
            query.append(self.attn_anchor(self._anchors_hid[label]))
            anchors.append(self._anchors_hid[label])
        query = torch.cat(query, 0)
        query = torch.tanh(query + self.attn_query(self._h[-1]))
        query = self.v_attn(query)
        if self.tanh_constant is not None:
            query = self.tanh_constant * torch.tanh(query)

        if mutable.n_chosen is None:
            logit = torch.cat([-query, query], 1)  # pylint: disable=invalid-unary-operand-type

            skip = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip_prob = torch.sigmoid(logit)
            kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
            self.sample_skip_penalty += kl
            log_prob = self.cross_entropy_loss(logit, skip)
            self._inputs = (torch.matmul(skip.float(), torch.cat(anchors, 0)) / (1. + torch.sum(skip))).unsqueeze(0)
        else:
            assert mutable.n_chosen == 1, "Input choice must select exactly one or any in ENAS."
            logit = query.view(1, -1)
            index = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip = F.one_hot(index, num_classes=mutable.n_candidates).view(-1)
            log_prob = self.cross_entropy_loss(logit, index)
            self._inputs = anchors[index.item()]

        self.sample_log_prob += torch.sum(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += torch.sum(entropy)
        return skip.bool()
