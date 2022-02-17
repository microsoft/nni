# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import torch
import torch.nn as nn
import torch.optim as optim

from nni.retiarii.nn.pytorch.api import LayerChoice, InputChoice
from .random import PathSamplingLayerChoice, PathSamplingInputChoice
from .base_lightning import BaseOneShotLightningModule
from .enas import ReinforceController, ReinforceField


class EnasModule(BaseOneShotLightningModule):
    """
    The ENAS module. There are 2 steps in an epoch. 1: training model parameters. 2: training ENAS RL agent. The agent will produce
    a sample of model architecture to get the best reward.
    The ENASModule should be trained with :class:`nni.retiarii.oneshot.utils.ConcatenateTrainValDataloader`.

    Parameters
    ----------
    base_model : pl.LightningModule
        he evaluator in ``nni.retiarii.evaluator.lightning``. User defined model is wrapped by base_model, and base_model will
        be wrapped by this model.
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
        Gradient clipping value.
    custom_replace_dict : Dict[Type[nn.Module], Callable[[nn.Module], nn.Module]], default = None
        The custom xxxChoice replace method. Keys should be xxxChoice type and values should return an ``nn.module``. This custom
        replace dict will override the default replace dict of each NAS method.

    Reference
    ----------
    .. [enas] H. Pham, M. Guan, B. Zoph, Q. Le, and J. Dean, “Efficient Neural Architecture Search via Parameters Sharing,”
        in Proceedings of the 35th International Conference on Machine Learning, Jul. 2018, pp. 4095-4104.
        Available: https://proceedings.mlr.press/v80/pham18a.html
    """
    def __init__(self, base_model, ctrl_kwargs = None,
                 entropy_weight = 1e-4, skip_weight = .8, baseline_decay = .999,
                 ctrl_steps_aggregate = 20, grad_clip = 0, custom_replace_dict = None):
        super().__init__(base_model, custom_replace_dict)

        self.nas_fields = [ReinforceField(name, len(module),
                                          isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1)
                           for name, module in self.nas_modules]
        self.controller = ReinforceController(self.nas_fields, **(ctrl_kwargs or {}))

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
        # The ConcatenateTrainValDataloader yields both data and which dataloader it comes from.
        batch, source = batch

        if source == 'train':
            # step 1: train model params
            self._resample()
            self.call_user_optimizers('zero_grad')
            loss_and_metrics = self.model.training_step(batch, batch_idx)
            w_step_loss = loss_and_metrics['loss'] \
                if isinstance(loss_and_metrics, dict) else loss_and_metrics
            self.manual_backward(w_step_loss)
            self.call_user_optimizers('step')
            return loss_and_metrics

        if source == 'val':
            # step 2: train ENAS agent
            x, y = batch
            arc_opt = self.architecture_optimizers
            arc_opt.zero_grad()
            self._resample()
            with torch.no_grad():
                logits = self.model(x)
            # use the default metric of self.model as reward function
            if len(self.model.metrics) == 1:
                _, metric = next(iter(self.model.metrics.items()))
            else:
                if 'default' not in self.model.metrics.keys():
                    raise KeyError('model.metrics should contain a ``default`` key when' \
                        'there are multiple metrics')
                metric = self.model.metrics['default']

            reward = metric(logits, y)
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

    def _resample(self):
        """
        Resample the architecture as ENAS result. This doesn't require an ``export`` method in nas_modules to work.
        """
        result = self.controller.resample()
        for name, module in self.nas_modules:
            module.sampled = result[name]

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()


class RandomSampleModule(BaseOneShotLightningModule):
    """
    Random Sampling NAS Algorithm. In each epoch, model parameters are trained after a uniformly random sampling of each choice.
    The training result is also a random sample of the search space.

    Parameters
    ----------
    base_model : pl.LightningModule
        he evaluator in ``nni.retiarii.evaluator.lightning``. User defined model is wrapped by base_model, and base_model will
        be wrapped by this model.
    custom_replace_dict : Dict[Type[nn.Module], Callable[[nn.Module], nn.Module]], default = None
        The custom xxxChoice replace method. Keys should be xxxChoice type and values should return an ``nn.module``. This custom
        replace dict will override the default replace dict of each NAS method.
    """
    automatic_optimization = True

    def training_step(self, batch, batch_idx):
        self._resample()
        return self.model.training_step(batch, batch_idx)

    @property
    def default_replace_dict(self):
        return {
            LayerChoice : PathSamplingLayerChoice,
            InputChoice : PathSamplingInputChoice
        }

    def _resample(self):
        """
        Resample the architecture as RandomSample result. This is simply a uniformly sampling that doesn't require an ``export``
        method in nas_modules to work.
        """
        result = {}
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = random.randint(0, len(module) - 1)
            module.sampled = result[name]
        return result

    def export(self):
        return self._resample()
