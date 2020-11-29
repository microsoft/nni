# FIXME: WIP

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .mutator import EnasMutator
from nni.nas.pytorch.utils import AverageMeterGroup, to_device
from nni.nas.pytorch.trainer import Trainer
import torch.optim as optim
from itertools import cycle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .random import PathSamplingLayerChoice, PathSamplingInputChoice
from .utils import AverageMeterGroup, replace_layer_choice, replace_input_choice


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


class ReinforceMutator(nn.Module):
    """
    A mutator that mutates the graph with RL.

    Parameters
    ----------
    modules : list of nn.Module
        Module list that contains all mutable modules. 
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

    def __init__(self, modules, lstm_size=64, lstm_num_layers=1, tanh_constant=1.5, cell_exit_extra_step=False,
                 skip_target=0.4, temperature=None, branch_bias=0.25, entropy_reduction='sum'):
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
        self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]),
                                         requires_grad=False)  # pylint: disable=not-callable
        assert entropy_reduction in ['sum', 'mean'], 'Entropy reduction must be one of sum and mean.'
        self.entropy_reduction = torch.sum if entropy_reduction == 'sum' else torch.mean
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.bias_dict = nn.ParameterDict()

        self.max_layer_choice = 0
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                if self.max_layer_choice == 0:
                    self.max_layer_choice = len(mutable)
                assert self.max_layer_choice == len(mutable), \
                    'ENAS mutator requires all layer choice have the same number of candidates.'
                # We are judging by keys and module types to add biases to layer choices. Needs refactor.
                if 'reduce' in mutable.key:
                    def is_conv(choice):
                        return 'conv' in str(type(choice)).lower()
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
            assert mutable.n_chosen == 1, 'Input choice must select exactly one or any in ENAS.'
            logit = query.view(1, -1)
            index = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip = F.one_hot(index, num_classes=mutable.n_candidates).view(-1)
            log_prob = self.cross_entropy_loss(logit, index)
            self._inputs = anchors[index.item()]

        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
        return skip.bool()


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


logger = logging.getLogger(__name__)


class EnasTrainer(Trainer):
    """
    ENAS trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    reward_function : callable
        Receives logits and ground truth label, return a tensor, which will be feeded to RL controller as reward.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset_train : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    dataset_valid : Dataset
        Dataset for testing.
    mutator : EnasMutator
        Use when customizing your own mutator or a mutator with customized parameters.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    callbacks : list of Callback
        list of callbacks to trigger at events.
    entropy_weight : float
        Weight of sample entropy loss.
    skip_weight : float
        Weight of skip penalty loss.
    baseline_decay : float
        Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
    child_steps : int
        How many mini-batches for model training per epoch.
    mutator_lr : float
        Learning rate for RL controller.
    mutator_steps_aggregate : int
        Number of steps that will be aggregated into one mini-batch for RL controller.
    mutator_steps : int
        Number of mini-batches for each epoch of RL controller learning.
    aux_weight : float
        Weight of auxiliary head loss. ``aux_weight * aux_loss`` will be added to total loss.
    """

    def __init__(self, model, loss, metrics, reward_function,
                 optimizer, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None, callbacks=None,
                 entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999,
                 mutator_lr=0.00035, mutator_steps_aggregate=20, aux_weight=0.4):
        super().__init__(model, mutator if mutator is not None else EnasMutator(model),
                         loss, metrics, optimizer, num_epochs, dataset_train, dataset_valid,
                         batch_size, workers, device, log_frequency, callbacks)
        self.reward_function = reward_function
        self.mutator_optim = optim.Adam(self.mutator.parameters(), lr=mutator_lr)
        self.batch_size = batch_size
        self.workers = workers

        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.
        self.mutator_steps_aggregate = mutator_steps_aggregate
        self.mutator_steps = mutator_steps
        self.child_steps = child_steps
        self.aux_weight = aux_weight
        self.test_arc_per_epoch = test_arc_per_epoch

        self.init_dataloader()

    def init_dataloader(self):
        n_train = len(self.dataset_train)
        split = n_train // 10
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:-split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[-split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=self.batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=self.workers)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=self.batch_size,
                                                       num_workers=self.workers)
        self.train_loader = cycle(self.train_loader)
        self.valid_loader = cycle(self.valid_loader)

    def train_one_epoch(self, epoch):
        # Sample model and train
        self.model.train()
        self.mutator.eval()
        meters = AverageMeterGroup()
        for step in range(1, self.child_steps + 1):
            x, y = next(self.train_loader)
            x, y = to_device(x, self.device), to_device(y, self.device)
            self.optimizer.zero_grad()

            with torch.no_grad():
                self.mutator.reset()
            self._write_graph_status()
            logits = self.model(x)

            if isinstance(logits, tuple):
                logits, aux_logits = logits
                aux_loss = self.loss(aux_logits, y)
            else:
                aux_loss = 0.
            metrics = self.metrics(logits, y)
            loss = self.loss(logits, y)
            loss = loss + self.aux_weight * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            metrics["loss"] = loss.item()
            meters.update(metrics)

            if self.log_frequency is not None and step % self.log_frequency == 0:
                logger.info("Model Epoch [%d/%d] Step [%d/%d]  %s", epoch + 1,
                            self.num_epochs, step, self.child_steps, meters)

        # Train sampler (mutator)
        self.model.eval()
        self.mutator.train()
        meters = AverageMeterGroup()
        for mutator_step in range(1, self.mutator_steps + 1):
            self.mutator_optim.zero_grad()
            for step in range(1, self.mutator_steps_aggregate + 1):
                x, y = next(self.valid_loader)
                x, y = to_device(x, self.device), to_device(y, self.device)

                self.mutator.reset()
                with torch.no_grad():
                    logits = self.model(x)
                self._write_graph_status()
                metrics = self.metrics(logits, y)
                reward = self.reward_function(logits, y)
                if self.entropy_weight:
                    reward += self.entropy_weight * self.mutator.sample_entropy.item()
                self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
                loss = self.mutator.sample_log_prob * (reward - self.baseline)
                if self.skip_weight:
                    loss += self.skip_weight * self.mutator.sample_skip_penalty
                metrics["reward"] = reward
                metrics["loss"] = loss.item()
                metrics["ent"] = self.mutator.sample_entropy.item()
                metrics["log_prob"] = self.mutator.sample_log_prob.item()
                metrics["baseline"] = self.baseline
                metrics["skip"] = self.mutator.sample_skip_penalty

                loss /= self.mutator_steps_aggregate
                loss.backward()
                meters.update(metrics)

                cur_step = step + (mutator_step - 1) * self.mutator_steps_aggregate
                if self.log_frequency is not None and cur_step % self.log_frequency == 0:
                    logger.info("RL Epoch [%d/%d] Step [%d/%d] [%d/%d]  %s", epoch + 1, self.num_epochs,
                                mutator_step, self.mutator_steps, step, self.mutator_steps_aggregate,
                                meters)

            nn.utils.clip_grad_norm_(self.mutator.parameters(), 5.)
            self.mutator_optim.step()

    def validate_one_epoch(self, epoch):
        with torch.no_grad():
            for arc_id in range(self.test_arc_per_epoch):
                meters = AverageMeterGroup()
                for x, y in self.test_loader:
                    x, y = to_device(x, self.device), to_device(y, self.device)
                    self.mutator.reset()
                    logits = self.model(x)
                    if isinstance(logits, tuple):
                        logits, _ = logits
                    metrics = self.metrics(logits, y)
                    loss = self.loss(logits, y)
                    metrics["loss"] = loss.item()
                    meters.update(metrics)

                logger.info("Test Epoch [%d/%d] Arc [%d/%d] Summary  %s",
                            epoch + 1, self.num_epochs, arc_id + 1, self.test_arc_per_epoch,
                            meters.summary())
