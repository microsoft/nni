# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ReinforceField:
    """
    A field with ``name``, with ``total`` choices. ``choose_one`` is true if one and only one is meant to be
    selected. Otherwise, any number of choices can be chosen.
    """

    def __init__(self, name, total, choose_one):
        self.name = name
        self.total = total
        self.choose_one = choose_one

    def __repr__(self):
        return f'ReinforceField(name={self.name}, total={self.total}, choose_one={self.choose_one})'


class ReinforceController(nn.Module):
    """
    A controller that mutates the graph with RL.

    Parameters
    ----------
    fields : list of ReinforceField
        List of fields to choose.
    lstm_size : int
        Controller LSTM hidden units.
    lstm_num_layers : int
        Number of layers for stacked LSTM.
    tanh_constant : float
        Logits will be equal to ``tanh_constant * tanh(logits)``. Don't use ``tanh`` if this value is ``None``.
    skip_target : float
        Target probability that skipconnect (chosen by InputChoice) will appear.
        If the chosen number of inputs is away from the ``skip_connect``, there will be
        a sample skip penalty which is a KL divergence added.
    temperature : float
        Temperature constant that divides the logits.
    entropy_reduction : str
        Can be one of ``sum`` and ``mean``. How the entropy of multi-input-choice is reduced.
    """

    def __init__(self, fields, lstm_size=64, lstm_num_layers=1, tanh_constant=1.5,
                 skip_target=0.4, temperature=None, entropy_reduction='sum'):
        super(ReinforceController, self).__init__()
        self.fields = fields
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.skip_target = skip_target

        self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)
        self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1)
        self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]),  # pylint: disable=not-callable
                                         requires_grad=False)
        assert entropy_reduction in ['sum', 'mean'], 'Entropy reduction must be one of sum and mean.'
        self.entropy_reduction = torch.sum if entropy_reduction == 'sum' else torch.mean
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.soft = nn.ModuleDict({
            field.name: nn.Linear(self.lstm_size, field.total, bias=False) for field in fields
        })
        self.embedding = nn.ModuleDict({
            field.name: nn.Embedding(field.total, self.lstm_size) for field in fields
        })

    def resample(self, return_prob=False):
        self._initialize()
        result = dict()
        for field in self.fields:
            result[field.name] = self._sample_single(field, return_prob=return_prob)
        return result

    def _initialize(self):
        self._inputs = self.g_emb.data
        self._c = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self._h = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self.sample_log_prob: torch.Tensor = cast(torch.Tensor, 0)
        self.sample_entropy: torch.Tensor = cast(torch.Tensor, 0)
        self.sample_skip_penalty: torch.Tensor = cast(torch.Tensor, 0)

    def _lstm_next_step(self):
        self._h, self._c = self.lstm(self._inputs, (self._h, self._c))

    def _sample_single(self, field, return_prob):
        self._lstm_next_step()
        logit = self.soft[field.name](self._h[-1])
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        if field.choose_one:
            sampled_dist = F.softmax(logit, dim=-1)
            sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            log_prob = self.cross_entropy_loss(logit, sampled)
            self._inputs = self.embedding[field.name](sampled)
        else:
            sampled_dist = torch.sigmoid(logit)
            logit = logit.view(-1, 1)
            logit = torch.cat([-logit, logit], 1)  # pylint: disable=invalid-unary-operand-type
            sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip_prob = torch.sigmoid(logit)
            kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
            self.sample_skip_penalty += kl
            log_prob = self.cross_entropy_loss(logit, sampled)
            sampled = sampled.nonzero().view(-1)
            if sampled.sum().item():
                self._inputs = (torch.sum(self.embedding[field.name](sampled.view(-1)), 0) / (1. + torch.sum(sampled))).unsqueeze(0)
            else:
                self._inputs = torch.zeros(1, self.lstm_size, device=self.embedding[field.name].weight.device)  # type: ignore

        sampled = sampled.detach().cpu().numpy().tolist()
        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
        if len(sampled) == 1:
            sampled = sampled[0]

        if return_prob:
            return sampled_dist.flatten().detach().cpu().numpy().tolist()
        return sampled
