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
        prev_h, prev_c = hidden
        next_h, next_c = [], []
        for i, m in enumerate(self.lstm_modules):
            curr_h, curr_c = m(inputs, (prev_h[i], prev_c[i]))
            next_c.append(curr_c)
            next_h.append(curr_h)
            # current implementation only supports batch size equals 1,
            # but the algorithm does not necessarily have this limitation
            inputs = curr_h[-1].view(1, -1)
        return next_h, next_c


class EnasMutator(Mutator):
    """
    A mutator that mutates the graph with RL.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    lstm_size : int
        Controller LSTM hidden units.
    lstm_num_layers : int
        Number of layers for stacked LSTM.
    tanh_constant : float
        Logits will be equal to ``tanh_constant * tanh(logits)``. Don't use ``tanh`` if this value is ``None``.
    cell_exit_extra_step : bool
        If true, RL controller will perform an extra step at the exit of each MutableScope, dump the hidden state
        and mark it as the hidden state of this MutableScope. This is to align with the original implementation of paper.
    skip_target : float
        Target probability that skipconnect will appear.
    temperature : float
        Temperature constant that divides the logits.
    branch_bias : float
        Manual bias applied to make some operations more likely to be chosen.
        Currently this is implemented with a hardcoded match rule that aligns with original repo.
        If a mutable has a ``reduce`` in its key, all its op choices
        that contains `conv` in their typename will receive a bias of ``+self.branch_bias`` initially; while others
        receive a bias of ``-self.branch_bias``.
    entropy_reduction : str
        Can be one of ``sum`` and ``mean``. How the entropy of multi-input-choice is reduced.
    """

    def __init__(self, model, lstm_size=64, lstm_num_layers=1, tanh_constant=1.5, cell_exit_extra_step=False,
                 skip_target=0.4, temperature=None, branch_bias=0.25, entropy_reduction="sum"):
        super().__init__(model)
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.cell_exit_extra_step = cell_exit_extra_step
        self.skip_target = skip_target
        self.branch_bias = branch_bias

        self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)
        self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1)
        self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]), requires_grad=False)  # pylint: disable=not-callable
        assert entropy_reduction in ["sum", "mean"], "Entropy reduction must be one of sum and mean."
        self.entropy_reduction = torch.sum if entropy_reduction == "sum" else torch.mean
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.bias_dict = nn.ParameterDict()

        self.max_layer_choice = 0
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                if self.max_layer_choice == 0:
                    self.max_layer_choice = len(mutable)
                assert self.max_layer_choice == len(mutable), \
                    "ENAS mutator requires all layer choice have the same number of candidates."
                # We are judging by keys and module types to add biases to layer choices. Needs refactor.
                if "reduce" in mutable.key:
                    def is_conv(choice):
                        return "conv" in str(type(choice)).lower()
                    bias = torch.tensor([self.branch_bias if is_conv(choice) else -self.branch_bias  # pylint: disable=not-callable
                                         for choice in mutable])
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
        self._h, self._c = self.lstm(self._inputs, (self._h, self._c))

    def _mark_anchor(self, key):
        self._anchors_hid[key] = self._h[-1]

    def _sample_layer_choice(self, mutable):
        self._lstm_next_step()
        logit = self.soft(self._h[-1])
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        if mutable.key in self.bias_dict:
            logit += self.bias_dict[mutable.key]
        branch_id = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
        log_prob = self.cross_entropy_loss(logit, branch_id)
        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
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
        if self.temperature is not None:
            query /= self.temperature
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

        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
        return skip.bool()
