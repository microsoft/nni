# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from nni.retiarii.nn.pytorch.api import LayerChoice, InputChoice

from .random import PathSamplingLayerChoice, PathSamplingInputChoice
from .base_lightning import BaseOneShotLightningModule


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
        Target probability that skipconnect will appear.
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

    def resample(self):
        self._initialize()
        result = dict()
        for field in self.fields:
            result[field.name] = self._sample_single(field)
        return result

    def _initialize(self):
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

    def _sample_single(self, field):
        self._lstm_next_step()
        logit = self.soft[field.name](self._h[-1])
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        if field.choose_one:
            sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            log_prob = self.cross_entropy_loss(logit, sampled)
            self._inputs = self.embedding[field.name](sampled)
        else:
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
                self._inputs = torch.zeros(1, self.lstm_size, device=self.embedding[field.name].weight.device)

        sampled = sampled.detach().numpy().tolist()
        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
        if len(sampled) == 1:
            sampled = sampled[0]
        return sampled


class EnasModule(BaseOneShotLightningModule):
    """
    The ENAS Model. In each epoch, model parameters are first trained with weight sharing, followed by
    the enas RL agent. The agent will produce a sample of model architecture, and the reward function
    is used to train the agent.
    The ENAS Model should be trained with ConcatenateTraiValDataloader in nn.retiarii.oneshot.pytorch.utils.
    See base class for more attributes.

    Reference
    ----------
    .. [enas] H. Pham, M. Guan, B. Zoph, Q. Le, and J. Dean, “Efficient Neural Architecture Search via Parameters Sharing,”
        in Proceedings of the 35th International Conference on Machine Learning, Jul. 2018, pp. 4095-4104.
        Available: https://proceedings.mlr.press/v80/pham18a.html
    """

    automatic_optimization = False

    def __init__(self, base_model, reward_function, ctrl_kwargs = None,
                 entropy_weight = 1e-4, skip_weight = .8, baseline_decay = .999,
                 ctrl_steps_aggregate = 20, grad_clip = 0, custom_replace_dict = None):
        """
        Parameters
        ----------
        base_model : pl.LightningModule
            The module in evaluators in nni.retiarii.evaluator.lightning. User defined model
            is wrapped by ''base_model'', and ''base_model'' will be wrapped by this model.
        reward_function : callable
            Receives logits and ground truth label, return a tensor, which will be feeded to RL controller as reward.
        ctrl_kwargs : dict
            Optional kwargs that will be passed to :class:`ReinforceController`.
        entropy_weight : float
            Weight of sample entropy loss.
        skip_weight : float
            Weight of skip penalty loss.
        baseline_decay : float
            Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
        ctrl_steps_aggregate : int
            Number of steps that will be aggregated into one mini-batch for RL controller.
        grad_clip : float
            Gradient clipping vlaue
        custom_replace_dict : Dict[Type[nn.Module], Callable[[nn.Module], nn.Module]], default = None
            The custom xxxChoice replace method. Keys should be xxxChoice type and values should
            return an nn.module. This custom replace dict will override the default replace
            dict of each NAS method.
        """
        super().__init__(base_model, custom_replace_dict)

        self.nas_fields = [ReinforceField(name, len(module),
                                          isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1)
                           for name, module in self.nas_modules]
        self.controller = ReinforceController(self.nas_fields, **(ctrl_kwargs or {}))

        self.reward_function = reward_function

        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.
        self.ctrl_steps_aggregate = ctrl_steps_aggregate
        self.grad_clip = grad_clip

    def configure_architecture_optimizers(self):
        return optim.Adam(self.controller.parameters(), lr=3.5e-4)

    @property
    def default_replace_dict(self):
        return {
            LayerChoice : PathSamplingLayerChoice,
            InputChoice : PathSamplingInputChoice
        }

    def training_step(self, batch, batch_idx):
        # grad manually
        opts = self.optimizers()
        arc_opt = opts[0]
        w_opt = opts[1:]

        # the batch is composed with x, y, b, where b denote if the data provided
        # is from the training set
        batch, is_train_batch = batch

        if is_train_batch:
            # step 1: train model params
            self._resample()
            for opt in w_opt:
                opt.zero_grad()
            w_step_loss = self.model.training_step(batch, batch_idx)
            self.manual_backward(w_step_loss)
            for opt in w_opt:
                opt.step()
        else:
            # step 2: train ENAS agent
            x, y = batch
            arc_opt.zero_grad()
            self._resample()
            with torch.no_grad():
                logits = self.model(x)
            reward = self.model.metrics['acc'](logits, y) # reward_function(logits, y)

            if self.entropy_weight:
                reward = reward + self.entropy_weight * self.controller.sample_entropy.item()
            self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
            rnn_step_loss = self.controller.sample_log_prob * (reward - self.baseline)
            if self.skip_weight:
                rnn_step_loss = rnn_step_loss + self.skip_weight * self.controller.sample_skip_penalty

            rnn_step_loss = rnn_step_loss / self.ctrl_steps_aggregate
            self.manual_backward(rnn_step_loss)

            if (batch_idx + 1) % self.ctrl_steps_aggregate == 0:
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip)
                arc_opt.step()
                arc_opt.zero_grad()


    def validation_step(self, batch, batch_idx):
        pass

    def _resample(self):
        result = self.controller.resample()
        for name, module in self.nas_modules:
            module.sampled = result[name]

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()



class RandomSampleModule(BaseOneShotLightningModule):
    """
    Random Sampling NAS Algorithm. In each epoch, model parameters are first trained after a uniformly random
    sampling of each choice. The training result is also a random sample of search space.
    The RandomSample Model should be trained with ConcatenateTraiValDataloader in nn.retiarii.oneshot.pytorch.utils.
    See base class for more attributes.
    """
    def __init__(self, base_model, custom_replace_dict = None):
        """
        Parameters
        ----------
        base_model : pl.LightningModule
            The module in evaluators in nni.retiarii.evaluator.lightning. User defined model
            is wrapped by ''base_model'', and ''base_model'' will be wrapped by this model.
        custom_replace_dict : Dict[Type[nn.Module], Callable[[nn.Module], nn.Module]], default = None
            The custom xxxChoice replace method. Keys should be xxxChoice type and values should
            return an nn.module. This custom replace dict will override the default replace
            dict of each NAS method.
        """
        super().__init__(base_model, custom_replace_dict)

    def training_step(self, batch, batch_idx):
        self._resample()
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._resample()
        return self.model.validation_step(batch, batch_idx)

    @property
    def default_replace_dict(self):
        return {
            LayerChoice : PathSamplingLayerChoice,
            InputChoice : PathSamplingInputChoice
        }

    def _resample(self):
        # The simplest sampling-based NAS method.
        # Each NAS module is uniformly sampled.
        result = {}
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = random.randint(0, len(module) - 1)
            module.sampled = result[name]
        return result

    def export(self):
        return self._resample()

