# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import warnings
from typing import Optional, TYPE_CHECKING

from nni.mutable import SampleValidationError
from nni.nas.execution import ExecutionEngine
from nni.nas.space import ExecutableModelSpace

from .base import Strategy

try:
    has_tianshou = True
    from tianshou.data import ReplayBuffer
    from ._rl_impl import PolicyFactory, TuningTrajectoryGenerator, default_policy_fn
except ImportError:
    has_tianshou = False

if TYPE_CHECKING:
    from tianshou.policy import BasePolicy


_logger = logging.getLogger(__name__)


class PolicyBasedRL(Strategy):
    """
    Algorithm for policy-based reinforcement learning.
    This is a wrapper of algorithms provided in tianshou (PPO by default),
    and can be easily customized with other algorithms that inherit ``BasePolicy``
    (e.g., `REINFORCE <https://link.springer.com/content/pdf/10.1007/BF00992696.pdf>`__
    as in `this paper <https://arxiv.org/abs/1611.01578>`__).

    Parameters
    ----------
    samples_per_update
        How many models (trajectories) each time collector collects.
        After each collect, trainer will sample batch from replay buffer and do the update.
    replay_buffer_size
        Size of replay buffer.
        If it's none, the size will be the expected trajectory length times ``samples_per_update``.
    reward_for_invalid
        The reward for a sample that didn't pass validation, or the training doesn't return a metric.
        If not provided, failed models will be simply ignored as if nothing happened.
    policy_fn
        Since environment is created on the fly, the policy needs to be a factory function that creates a policy on-the-fly.
        It takes :class:`~nni.nas.strategy._rl_impl.TuningEnvironment` as input and returns a policy.
        By default, it will use the policy returned by :func:`~nni.nas.strategy._rl_impl.default_policy_fn`.
    update_kwargs
        Keyword arguments for ``policy.update``. See tianshou's BasePolicy for details.
        There is a special key ``"update_times"`` that can be used to specify how many times ``policy.update`` is called,
        which can be used to sufficiently exploit the current available trajectories in the replay buffer
        (for example when actor and critic needs to be updated alternatively multiple times).
        By default, it's ``{'batch_size': 32, 'repeat': 5, 'update_times': 5}``.
    """

    _invalid_patience = 20

    def __init__(self, *, samples_per_update: int = 20,
                 replay_buffer_size: int | None = None,
                 reward_for_invalid: float | None = None,
                 policy_fn: Optional[PolicyFactory] = None,
                 update_kwargs: dict | None = None,
                 **kwargs):
        super().__init__()

        if 'max_collect' in kwargs:
            warnings.warn('`max_collect` is deprecated. It has no effect now.', DeprecationWarning)
        if 'trial_per_collect' in kwargs:
            warnings.warn('`trial_per_collect` is deprecated. Use `samples_per_update` instead.', DeprecationWarning)
            samples_per_update = kwargs['trial_per_collect']

        if not has_tianshou:
            raise ImportError('`tianshou` is required to run RL-based strategy. '
                              'Please use "pip install tianshou" to install it beforehand.')

        self.policy_fn = policy_fn or default_policy_fn
        self.samples_per_update = samples_per_update
        self.replay_buffer_size = replay_buffer_size
        self.reward_for_invalid = reward_for_invalid
        self.update_kwargs = {'batch_size': 32, 'repeat': 5, 'update_times': 5} if update_kwargs is None else update_kwargs

        self._current_episode = 0
        self._successful_episode = 0
        self._running_models: list[tuple[ExecutableModelSpace, TuningTrajectoryGenerator]] = []
        self._trajectory_count = 0
        self._policy: BasePolicy | None = None
        self._replay_buffer: ReplayBuffer | None = None

    def extra_repr(self) -> str:
        return f'samples_per_update={self.samples_per_update}, replay_buffer_size={self.replay_buffer_size}, ' + \
            f'reward_for_invalid={self.reward_for_invalid}'

    def _harvest_running_models(self) -> bool:
        """Harvest completed models and add their trajectories to replay buffer.

        Return true if the policy has just been updated on the latest buffer or no new trajectories found.
        False otherwise.
        """
        running_indices = []
        recently_updated = True
        for index, (model, generator) in enumerate(self._running_models):
            if model.status.completed():
                if model.metric is not None:
                    # No matter success or failure, as long as it is completed and gets a metric.
                    trajectory = generator.send_reward(model.metric)
                    _logger.info('[Trajectory %4d] (%s, %s) %s', self._trajectory_count + 1, model.status.value, model.metric, model.sample)
                    recently_updated = self._add_trajectory(trajectory)
                else:
                    _logger.info('%s has no metric. Skip.', model)
            else:
                running_indices.append(index)

        self._running_models = [self._running_models[i] for i in running_indices]
        return recently_updated

    def _add_trajectory(self, trajectory: ReplayBuffer) -> bool:
        """Add the trajectory to replay buffer and execute update if necessary.

        Return true if an update is just executed. False otherwise.
        """
        assert self._replay_buffer is not None
        self._replay_buffer.update(trajectory)
        self._trajectory_count += 1
        if self._trajectory_count % self.samples_per_update == 0:
            self._update_policy()
            return True
        return False

    def _update_policy(self) -> None:
        """Update the RL policy on current replay buffer."""
        _logger.info('[Trajectory %4d] Updating policy...', self._trajectory_count)
        assert self._policy is not None and self._replay_buffer is not None
        update_times = self.update_kwargs.get('update_times', 1)
        for _ in range(update_times):
            self._policy.update(0, self._replay_buffer, **self.update_kwargs)

    def _initialize(self, model_space: ExecutableModelSpace, engine: ExecutionEngine) -> ExecutableModelSpace:
        generator = TuningTrajectoryGenerator(model_space, self.policy_fn)
        self._policy = generator.policy

        if self.replay_buffer_size is None:
            replay_buffer_size = generator.expected_trajectory_length * self.samples_per_update
        else:
            replay_buffer_size = self.replay_buffer_size

        self._replay_buffer = ReplayBuffer(replay_buffer_size)

        return model_space

    def _run(self) -> None:
        assert self._policy is not None and self._replay_buffer is not None

        self._policy.train()

        _invalid_count = 0

        _logger.info('Sampling models with RL policy:\n%s', self._policy)

        while True:
            self._harvest_running_models()

            if not self.wait_for_resource():
                _logger.info('Budget exhausted. No more sampling.')
                break

            generator = TuningTrajectoryGenerator(self.model_space, self._policy)

            sample = generator.next_sample()
            try:
                model = self.model_space.freeze(sample)
                _invalid_count = 0
            except SampleValidationError:
                _logger.debug('Invalid sample generated. It will be handled following the setting of `reward_for_invalid`: %s', sample)
                _invalid_count += 1
                if _invalid_count > self._invalid_patience:
                    _logger.warning('Too many (over %d) invalid samples generated. No more sampling.', self._invalid_patience)
                    break
                if self.reward_for_invalid is not None:
                    trajectory = generator.send_reward(self.reward_for_invalid)
                    _logger.info('[Trajectory %4d] (invalid, %s) %s',
                                 self._trajectory_count + 1,
                                 self.reward_for_invalid if self.reward_for_invalid is not None else 'skip',
                                 sample)
                    self._add_trajectory(trajectory)
                continue

            # Now the model training is destined to happen.
            self._running_models.append((model, generator))
            self.engine.submit_models(model)

        _logger.info('Harvesting final running models.')
        # First use engine.wait to wait for models to finish training.
        self.engine.wait_models(*[model for model, _ in self._running_models])
        # Then put their metrics into buffer.
        if not self._harvest_running_models():
            # Train on final harvested models.
            _logger.info('Training on final harvested models.')
            self._update_policy()

    def state_dict(self) -> dict:
        result = {
            'current_episode': self._current_episode,
            'successful_episode': self._successful_episode,
            'trajectory_count': self._trajectory_count,
            'num_running_models': len(self._running_models),
        }
        if self._policy is None or self._replay_buffer is None:
            _logger.warning('State dict of policy and replay buffer is not saved because they are not initialized yet.')
        else:
            result['policy'] = self._policy.state_dict()
            result['replay_buffer'] = self._replay_buffer
        return result

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict.get('num_running_models', 0) > 0:
            _logger.warning('Loaded state dict has %d running models. They will be ignored.', state_dict['num_running_models'])

        self._current_episode = state_dict['current_episode']
        self._successful_episode = state_dict['successful_episode']
        self._trajectory_count = state_dict['trajectory_count']

        if self._policy is None or self._replay_buffer is None:
            _logger.warning('State dict of policy and replay buffer is not restored because they are not initialized yet.')
        elif 'policy' not in state_dict or 'replay_buffer' not in state_dict:
            _logger.warning('Policy and replay buffer is not restored because they are not found in saved in state_dict.')
        else:
            self._policy.load_state_dict(state_dict['policy'])
            self._replay_buffer = state_dict['replay_buffer']
