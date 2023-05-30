# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Sampling-based one-shot implementation."""

from __future__ import annotations
import warnings
import logging
from typing import Any, Callable, TYPE_CHECKING

import pytorch_lightning as pl
import torch

from nni.mutable import Sample
from nni.nas.nn.pytorch import ModelSpace

from .base_lightning import BaseOneShotLightningModule
from .profiler import ProfilerPenalty, ExpectationProfilerPenalty, SampleProfilerPenalty

try:
    has_tianshou = True
    from tianshou.data import ReplayBuffer
    from tianshou.policy import BasePolicy
    if TYPE_CHECKING:
        from nni.nas.strategy._rl_impl import PolicyFactory, TuningTrajectoryGenerator
except ImportError:
    has_tianshou = False


_logger = logging.getLogger(__name__)


class RandomSamplingLightningModule(BaseOneShotLightningModule):
    """Search implementation of :class:`~nni.nas.strategy.RandomOneShot`.

    Do not need extra preprocessing of dataloader.

    See Also
    --------
    nni.nas.strategy.RandomOneShot
    nni.nas.pytorch.oneshot.base_lightning.BaseOneShotLightningModule
    """

    _sampling_patience = 100  # number of resample before giving up
    _sampling_attempt = 0

    def __init__(self, training_module: pl.LightningModule, filter: Callable[[Sample], bool] | None = None):  # pylint: disable=redefined-builtin
        super().__init__(training_module)
        self.filter = filter

    # turn on automatic optimization because nothing interesting is going on here.
    @property
    def automatic_optimization(self) -> bool:
        return True

    def _repeat_until_valid(self, func: Callable[[], Sample]) -> Sample:
        self._sampling_attempt = 0
        while self._sampling_attempt < self._sampling_patience:
            self._sampling_attempt += 1
            sample = func()
            if self.filter is None or self.filter(sample):
                return sample
        raise RuntimeError('Failed to sample a valid architecture after {} attempts.'.format(self._sampling_patience))

    def resample(self):
        return self._repeat_until_valid(super().resample)

    def training_step(self, *args: Any, **kwargs: Any):
        self.resample()
        self.log('sampling_attempt', self._sampling_attempt)
        return self.training_module.training_step(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any):
        self.resample()
        return self.training_module.validation_step(*args, **kwargs)

    def export(self) -> dict[str, Any]:
        """
        Export of Random one-shot. It will return an arbitrary architecture.
        """
        warnings.warn(
            'Direct export from RandomOneShot returns an arbitrary architecture. '
            'Sampling the best architecture from this trained supernet is another search process. '
            'Users need to do another search based on the checkpoint of the one-shot strategy.',
            UserWarning
        )
        return super().export()


class EnasLightningModule(BaseOneShotLightningModule):
    """Sampling-based super-net training but using an RL agent to control the sampling.

    The default implementation for :class:`~nni.nas.strategy.ENAS`.

    See Also
    --------
    nni.nas.strategy.ENAS
    RandomSamplingLightningModule
    """

    def __init__(self,
                 training_module: pl.LightningModule,
                 *,
                 batches_per_update: float = 20,
                 log_prob_every_n_step: int = 10,
                 replay_buffer_size: int | None = None,
                 reward_metric_name: str | None = None,
                 policy_fn: PolicyFactory | None = None,
                 update_kwargs: dict | None = None,
                 warmup_epochs: int = 0,
                 penalty: ProfilerPenalty | None = None):
        super().__init__(training_module)

        if not has_tianshou:
            raise ImportError('ENAS requires tianshou to be installed.')

        if reward_metric_name is None:
            _logger.warning(
                'It is strongly recommended to have `reward_metric_name` specified. '
                'It should be one of the metrics logged in `self.log` in evaluator. '
                'Otherwise it will infer the reward based on certain rules.'
            )

        self.batches_per_update = batches_per_update
        self.replay_buffer_size = replay_buffer_size
        self.log_prob_every_n_step = log_prob_every_n_step
        self.reward_metric_name = reward_metric_name

        self.policy_fn = policy_fn
        self.update_kwargs = update_kwargs or {}
        self.warmup_epochs = warmup_epochs

        self.penalty = penalty

        self._policy: BasePolicy | None = None
        self._generator: TuningTrajectoryGenerator | None = None
        self._replay_buffer: ReplayBuffer | None = None
        self._trajectory_counter: int = 0

    @property
    def policy(self) -> BasePolicy:
        if self._policy is None:
            raise RuntimeError('Policy is not initialized yet.')
        return self._policy

    @property
    def generator(self) -> TuningTrajectoryGenerator:
        if self._generator is None:
            raise RuntimeError('Generator is not initialized yet.')
        return self._generator

    def set_model(self, model: ModelSpace) -> None:
        if not isinstance(model, ModelSpace):
            raise TypeError('ModelSpace is required for ENAS.')

        from nni.nas.strategy._rl_impl import TuningTrajectoryGenerator
        self._generator = TuningTrajectoryGenerator(model, self.policy_fn)

        if self.replay_buffer_size is None:
            # Refresh the replay buffer every time the model is updated.
            replay_buffer_size = self._generator.expected_trajectory_length * self.batches_per_update
        else:
            replay_buffer_size = self.replay_buffer_size

        self._replay_buffer = ReplayBuffer(replay_buffer_size)
        self._policy = self._generator.policy

        return super().set_model(model)

    def training_step(self, batch_packed, batch_idx):
        if len(batch_packed) == 2:
            # Legacy (pytorch-lightning 1.x): The received batch is a tuple of (data, "train" | "val")
            batch, mode = batch_packed
        else:
            # New (pytorch-lightning 2.0+): a tuple of data, batch_idx, and dataloader_idx
            batch, _, dataloader_idx = batch_packed
            mode = 'train' if dataloader_idx == 0 else 'val'

        assert self._replay_buffer is not None

        step_output = None  # Sometimes step can be skipped

        if mode == 'train':
            # train model params
            self.policy.eval()
            self.resample()
            step_output = self.training_module.training_step(batch, batch_idx)
            w_step_loss = step_output['loss'] if isinstance(step_output, dict) else step_output
            self.advance_optimization(w_step_loss, batch_idx)

        elif self.warmup_epochs == 0 or self.trainer.current_epoch >= self.warmup_epochs:
            # Run a sample to retrieve the reward
            self.policy.train()
            sample = self.resample()
            step_output = self.training_module.validation_step(batch, batch_idx)

            # use the default metric of self.training_module as reward function
            if len(self.trainer.callback_metrics) == 1:
                _, metric = next(iter(self.trainer.callback_metrics.items()))
            else:
                metric_name = self.reward_metric_name or 'default'
                if metric_name not in self.trainer.callback_metrics:
                    raise KeyError(f'Model reported metrics should contain a ``{metric_name}`` key but '
                                   f'found multiple (or zero) metrics without default: {list(self.trainer.callback_metrics.keys())}. '
                                   f'Please try to set ``reward_metric_name`` to be one of the keys listed above. '
                                   f'If it is not working use self.log to report metrics with the specified key ``{metric_name}`` '
                                   'in validation_step, and remember to set on_step=True.')
                metric = self.trainer.callback_metrics[metric_name]
            reward: float = metric.item()

            if self.penalty is not None:
                if isinstance(self.penalty, SampleProfilerPenalty):
                    reward, details = self.penalty(reward, sample)
                elif isinstance(self.penalty, ExpectationProfilerPenalty):
                    reward, details = self.penalty(reward, self.export_probs())
                else:
                    raise TypeError(f'Unknown penalty type: {type(self.penalty)}')
                self.log_dict({f'penalty/{k}': v for k, v in details.items()})

            # Put it into replay buffer
            trajectory = self.generator.send_reward(reward)
            self._replay_buffer.update(trajectory)
            self._trajectory_counter += 1

        self.advance_lr_schedulers(batch_idx)

        if self._trajectory_counter > 0 and self._trajectory_counter % self.log_prob_every_n_step == 0:
            self.log_probs(self.export_probs())

        if self._trajectory_counter > 0 and self._trajectory_counter % self.batches_per_update == 0:
            # Export could be just called.
            # The policy must be in train mode to make update work.
            self.policy.train()
            update_times = self.update_kwargs.get('update_times', 1)
            for _ in range(update_times):
                self.policy.update(0, self._replay_buffer, **self.update_kwargs)

        return step_output

    def resample(self) -> Sample:
        """Resample the architecture with ENAS controller."""
        # policy could be either eval or train, depending on where you call it.
        with torch.no_grad():
            sample = self.generator.next_sample()
        for module in self.supernet_modules():
            module.resample(memo=sample)
        return sample

    def export_probs(self) -> Sample:
        """Export the probability from ENAS controller directly."""
        self.policy.eval()
        if self.generator.sample_logits is None:
            with torch.no_grad():
                self.generator.next_sample()
        assert self.generator.sample_logits is not None

        # Convert {'a': [0.2, 0.5, 0.3]} to {'a': {'choice0': 0.2, 'choice1': 0.5, 'choice2': 0.3} }
        return {
            label: dict(zip(self.generator.simplified_space[label].values, logits))
            for label, logits in self.generator.sample_logits.items()
        }

    def export(self):
        """Run one more inference of ENAS controller."""
        self.policy.eval()
        with torch.no_grad():
            return self.generator.next_sample()
