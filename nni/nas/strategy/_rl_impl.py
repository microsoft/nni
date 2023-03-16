# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# This file might cause import error for those who didn't install RL-related dependencies

"""The RL implementation for whoever wants to use RL to search for configurations in a space.

The implementation should be agnostic to space types (HPO / NAS) or strategy types (multi-trial / one-shot).

We assume that you have basic knowledge on how tianshou works,
including what is a replay buffer, how to train a tianshou policy.

The usage goes like (pseudo-code, won't run)::

    replay_buffer = ReplayBuffer(...)  # Please use ReplayBuffer instead of the vectorized one.

    generator = TuningTrajectoryGenerator(search_space)
    policy = generator.policy

    for i in range(100):
        sample = generator.next_sample()
        # evaluate the sample here
        trajectory = generator.send_reward(reward)
        replay_buffer.update(trajectory)  # Add the trajectory to the replay buffer

    policy.update(0, replay_buffer)

To use this implementation, tianshou and torch must be installed.

NOTE: We recommend creating the first policy with the generator.
The follow-up policies should reuse the first policy.
"""

from __future__ import annotations

__all__ = ['ObservationType', 'TuningEnvironment', 'TuningTrajectoryGenerator', 'PolicyFactory', 'default_policy_fn']

from copy import deepcopy
from typing import Tuple, Callable

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict

from gym import spaces
from tianshou.data import ReplayBuffer, Batch, to_torch, to_numpy
from tianshou.policy import BasePolicy, PPOPolicy

from nni.mutable import Categorical, CategoricalMultiple, Mutable, MutableAnnotation, Sample


class ObservationType(TypedDict):
    """The observation of :class:`TuningEnvironment`.

    Attributes
    ----------
    action_history
        The history of actions taken in the current trajectory.
    cur_step
        The current step of the trajectory.
    action_dim
        The number of choices in this step.
    """
    action_history: np.ndarray
    cur_step: int
    action_dim: int


EnvStepType = Tuple[ObservationType, float, bool, dict]


class TuningEnvironment(gym.Env[ObservationType, int]):
    """The evaluation environment for RL-based algorithms.

    Currently the environment mocks an environment of multiple choices.
    In each step, it returns an observation with the current action history and the number of choices in this step.
    The environment is "done" when the step count reaches the length of choices.

    The reward is always 0 as it's processed in :class:`TuningTrajectoryGenerator`.
    The environment isn't responsible for the actual configuration evaluation, neither will it invoke any training.

    It only supports search space with categorical choices for now.
    When there are multiple categorical choices with different number of choices,
    the environment will pad the choices to make them have the same length, and put the real number of choices in the observation.
    The policy returned by :func:`default_policy_fn` will ignore the padded choices.

    Environment is stateful. One environment instance should not be used in more than one trajectories at the same time.

    Parameters
    ----------
    num_choices
        The number of choices in each step.
    """

    def __init__(self, num_choices: list[int]):
        self.num_choices = num_choices
        self.max_num_choices = max(num_choices)
        self.num_steps = len(self.num_choices)

    @property
    def observation_space(self):
        return spaces.Dict({
            'action_history': spaces.MultiDiscrete([self.max_num_choices] * self.num_steps),
            'cur_step': spaces.Discrete(self.num_steps + 1),  # pad the step after end step
            'action_dim': spaces.Discrete(self.max_num_choices + 1)
        })

    @property
    def action_space(self):
        return spaces.Discrete(self.max_num_choices)

    def reset(self) -> tuple[ObservationType, dict]:
        self.action_history = np.zeros(self.num_steps, dtype=np.int32)
        self.cur_step = 0
        self.sample = {}
        return ObservationType(
            action_history=self.action_history,
            cur_step=self.cur_step,
            action_dim=self.num_choices[self.cur_step]
        ), {}

    def step(self, action: int) -> tuple[ObservationType, float, bool, bool, dict]:
        """Step the environment.

        Parameters
        ----------
        action
            Choice of the current step.
        """
        if action >= self.num_choices[self.cur_step]:
            raise ValueError(f'Current action {action} out of range {self.num_choices[self.cur_step]}.')
        self.action_history[self.cur_step] = action
        self.cur_step += 1
        obs: ObservationType = {
            'action_history': self.action_history,
            'cur_step': self.cur_step,
            'action_dim': self.num_choices[self.cur_step] if self.cur_step < self.num_steps else self.max_num_choices
        }

        if self.cur_step == self.num_steps:
            done = True
        else:
            done = False

        return obs, 0., done, False, {}


class TuningTrajectoryGenerator:
    """Generate a sample with the given search space and policy (factory) that is expected to work on the space.
    When the metric is ready, it returns the whole trajectory.

    The generator is single-threaded, meaning that it can't generate multiple samples,
    and wait for their rewards simultaneously.
    For parallelization, you can create multiple generators and share one policy.

    Parameters
    ----------
    policy
        The policy to use.
        For the first generator, as the environment is never created,
        it's recommended to use a factory.
        For the following generators, you can reuse the same policy.
    search_space
        The search space to search.

    Attributes
    ----------
    policy
        The policy it uses. Use the attribute as parameter of another generator to reuse the policy.
    env
        The environment it uses.
    sample
        The sample it just generated.
    sample_logits
        The logits of the sample.
    """

    policy: BasePolicy
    env: TuningEnvironment

    def __init__(self, search_space: Mutable, policy: PolicyFactory | BasePolicy | None = None) -> None:
        self.simplified_space: dict[str, Categorical | CategoricalMultiple] = {}
        self.search_labels: list[tuple[str, int | None]] = []
        self.num_choices: list[int] = []
        for label, mutable in search_space.simplify().items():
            if isinstance(mutable, MutableAnnotation):
                # Skip annotations like constraints.
                continue
            # Expand CategoricalMultiple to Categorical by default.
            if isinstance(mutable, CategoricalMultiple):
                if mutable.n_chosen is None:
                    for i in range(len(mutable.values)):
                        self.search_labels.append((label, i))
                        self.num_choices.append(2)  # true and false
                elif mutable.n_chosen == 1:
                    self.search_labels.append((label, None))
                    self.num_choices.append(len(mutable.values))
                else:
                    # TODO: support categorical multiple.
                    raise ValueError('CategoricalMultiple with n_chosen > 1 is not supported yet.')
                self.simplified_space[label] = mutable
            elif isinstance(mutable, Categorical):
                self.simplified_space[label] = mutable
                self.search_labels.append((label, None))
                # TODO: sampling priors are neglected.
                self.num_choices.append(len(mutable))
            else:
                raise ValueError('RL algorithms only supports Categorical for now.')
        self.env = TuningEnvironment(self.num_choices)

        if policy is None:
            self.policy = default_policy_fn(self.env)
        elif isinstance(policy, BasePolicy):
            self.policy = policy
        else:
            self.policy = policy(self.env)

        self._last_action: int | None = None
        self._trajectory: list[Batch] | None = None
        self._transition: Batch | None = None

        self.sample: Sample | None = None
        self.sample_logits: Sample | None = None

    @property
    def expected_trajectory_length(self) -> int:
        """Return the expected length of a trajectory."""
        return self.env.num_steps

    def next_sample(self) -> Sample:
        """Create a new trajectory, and return the sample when it's done.

        The sample isn't yet validated and thus can be possibly invalid.
        The caller needs to freeze the mutable with the sample by itself.

        The class will be in a state pending for reward after a call of :meth:`next_sample`.
        It will either receive the reward via :meth:`send_reward` or be reset via another :meth:`next_sample`.
        """
        obs, info = self.env.reset()
        last_state = None  # hidden state

        self._trajectory = []
        self._transition = Batch(
            obs=obs,  # only obs is set.
            act={},   # the rest are empty.
            rew={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info=info,
            policy={}
        )

        self.sample = {}
        self.sample_logits = {}

        step_count = 0

        while True:
            obs_batch = Batch([self._transition])    # the first dimension is batch-size
            policy_result = self.policy(obs_batch, last_state)
            # get bounded and remapped actions first (not saved into buffer)
            self._last_action = self.policy.map_action(to_numpy(policy_result.act))[0]
            # TODO: exploration noise

            # Update the sample and the logits.
            last_label, label_count = self.search_labels[step_count]
            mutable = self.simplified_space[last_label]
            if 'logits' in policy_result:
                logits = to_numpy(policy_result.logits)[0][:self.num_choices[step_count]].tolist()
            else:
                logits = None

            if label_count is None:
                # Categorical, or CategoryMultiple with n_chosen == 1.
                if isinstance(mutable, CategoricalMultiple):
                    self.sample[last_label] = [mutable.values[self._last_action]]
                else:
                    self.sample[last_label] = mutable.values[self._last_action]
                if logits is not None:
                    self.sample_logits[last_label] = logits
            else:
                # CategoryMultiple with n_chosen is None.
                if last_label not in self.sample:
                    self.sample[last_label] = []
                if self._last_action == 0:
                    self.sample[last_label].append(mutable.values[label_count])
                if logits is not None:
                    if last_label not in self.sample_logits:
                        self.sample_logits[last_label] = []
                    self.sample_logits[last_label].append(logits[0])

            self._transition.update(
                policy=self._canonicalize_policy_data(policy_result),
                act=to_numpy(policy_result.act)[0],  # the unmapped action
            )

            step_count += 1
            if step_count == len(self.num_choices):
                return self.sample

            obs_next, rew, terminated, truncated, info = self.env.step(self._last_action)
            assert not terminated, 'The environment should not be done yet.'

            self._transition.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=terminated,
                info=info
            )

            self._trajectory.append(deepcopy(self._transition))

            last_state = policy_result.get('state', None)
            self._transition.obs = self._transition.obs_next

    def send_reward(self, reward: float) -> ReplayBuffer:
        """Set the reward for the sample just created,
        and return the whole trajectory.

        Parameters
        ----------
        reward
            The reward for the sample just created.
            If None, the sample will be ignored.
        """

        assert self._trajectory is not None and self._transition is not None and self._last_action is not None

        obs_next, _, terminated, truncated, info = self.env.step(self._last_action)
        assert terminated, 'The environment should be done.'

        self._transition.update(
            obs_next=obs_next,
            rew=reward,
            terminated=terminated,
            truncated=truncated,
            done=terminated,
            info=info
        )
        self._trajectory.append(deepcopy(self._transition))

        # Return the trajectory as a replay buffer. Do conversion here.
        rv = ReplayBuffer(len(self._trajectory))
        for transition in self._trajectory:
            rv.add(transition)

        self._trajectory = self._transition = self._last_action = self.sample = self.sample_logits = None
        return rv

    def _canonicalize_policy_data(self, policy_result: Batch) -> dict:
        """Extract the "policy" part that is to be stored in the replay buffer.

        ``policy_result`` is returned from ``self.policy``. The batch must be of length 1.
        """
        policy_data = policy_result.get('policy', Batch())
        last_state = policy_result.get('state', None)
        if last_state is not None:
            policy_data.hidden_state = last_state

        if policy_data.is_empty():
            return policy_data
        else:
            # The batch must be of length 1.
            assert len(policy_data) == 1
            return policy_data[0]


PolicyFactory = Callable[[TuningEnvironment], BasePolicy]


def default_policy_fn(env: TuningEnvironment, lr: float = 1.0e-4, hidden_dim: int = 64) -> PPOPolicy:
    """Create a default policy for the given environment.

    The default policy is a PPO policy, with a simple LSTM + MLP network.

    To customize the parameters of this function, use::

        from functools import partial
        partial(default_policy_fn, lr=1.0e-3, hidden_dim=128)

    Parameters
    ----------
    lr
        Learning rate for the Adam optimizer.
    hidden_dim
        Hidden dimension of the LSTM and MLP.
    """
    net = Preprocessor(env.observation_space, hidden_dim)
    actor = Actor(env.action_space, net)
    critic = Critic(net)
    optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=lr)
    return PPOPolicy(actor, critic, optim, torch.distributions.Categorical,
                     discount_factor=1., action_space=env.action_space)


# The following are implementation of the default policy.

class Preprocessor(nn.Module):
    def __init__(self, obs_space, hidden_dim, num_layers=1):
        super().__init__()
        self.action_dim = obs_space['action_history'].nvec[0]
        self.step_dim = obs_space['action_history'].shape[0] + 2  # pad the start and end token
        self.hidden_dim = hidden_dim
        # first token is [SOS]
        self.embedding = nn.Embedding(self.action_dim + 1, hidden_dim)
        # self.rnn = nn.Linear(hidden_dim + self.step_dim, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim + self.step_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        batch_size = obs['action_history'].size(0)

        seq = F.pad(obs['action_history'] + 1, (1, 1))  # pad the start token and end token
        # end token is used to avoid out-of-range of v_s_. Will not actually affect BP.
        seq = self.embedding(seq.long())

        step_onehot = F.one_hot(torch.arange(self.step_dim, device=seq.device)).unsqueeze(0).repeat(batch_size, 1, 1)

        feature, _ = self.rnn(torch.cat((seq, step_onehot), -1))
        feature = feature[torch.arange(len(feature), device=feature.device), obs['cur_step'].long()]
        return self.fc(feature)


class Actor(nn.Module):
    def __init__(self, action_space, preprocess):
        super().__init__()
        self.preprocess = preprocess
        self.action_dim = action_space.n
        self.linear = nn.Linear(self.preprocess.hidden_dim, self.action_dim)

    def forward(self, obs, **kwargs):
        obs = to_torch(obs, device=self.linear.weight.device)
        out = self.linear(self.preprocess(obs))
        # to take care of choices with different number of options
        mask = torch.arange(self.action_dim, device=out.device).expand(len(out), self.action_dim) >= obs['action_dim'].unsqueeze(1)
        # NOTE: this could potentially be used for prior knowledge
        out_bias = torch.zeros_like(out)
        out_bias.masked_fill_(mask, float('-inf'))
        return F.softmax(out + out_bias, dim=-1), kwargs.get('state', None)


class Critic(nn.Module):
    def __init__(self, preprocess):
        super().__init__()
        self.preprocess = preprocess
        self.linear = nn.Linear(self.preprocess.hidden_dim, 1)

    def forward(self, obs, **kwargs):
        obs = to_torch(obs, device=self.linear.weight.device)
        return self.linear(self.preprocess(obs)).squeeze(-1)
