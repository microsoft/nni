# This file might cause import error for those who didn't install RL-related dependencies

import logging

import gym
import numpy as np
import torch
import torch.nn as nn

from gym import spaces
from tianshou.data import to_torch

from .utils import get_targeted_model
from ..graph import ModelStatus
from ..execution import submit_models, wait_models


_logger = logging.getLogger(__name__)


class ModelEvaluationEnv(gym.Env):
    def __init__(self, base_model, mutators, search_space):
        self.base_model = base_model
        self.mutators = mutators
        self.search_space = search_space
        self.ss_keys = list(self.search_space.keys())
        self.action_dim = max(map(lambda v: len(v), self.search_space.values()))
        self.num_steps = len(self.search_space)

    @property
    def observation_space(self):
        return spaces.Dict({
            'action_history': spaces.MultiDiscrete([self.action_dim] * self.num_steps),
            'cur_step': spaces.Discrete(self.num_steps + 1),
            'action_dim': spaces.Discrete(self.action_dim + 1)
        })

    @property
    def action_space(self):
        return spaces.Discrete(self.action_dim)

    def reset(self):
        self.action_history = np.zeros(self.num_steps, dtype=np.int32)
        self.cur_step = 0
        self.sample = {}
        return {
            'action_history': self.action_history,
            'cur_step': self.cur_step,
            'action_dim': len(self.search_space[self.ss_keys[self.cur_step]])
        }

    def step(self, action):
        cur_key = self.ss_keys[self.cur_step]
        assert action < len(self.search_space[cur_key]), \
            f'Current action {action} out of range {self.search_space[cur_key]}.'
        self.action_history[self.cur_step] = action
        self.sample[cur_key] = self.search_space[cur_key][action]
        self.cur_step += 1
        obs = {
            'action_history': self.action_history,
            'cur_step': self.cur_step,
            'action_dim': len(self.search_space[self.ss_keys[self.cur_step]]) \
                if self.cur_step < self.num_steps else self.action_dim
        }
        if self.cur_step == self.num_steps:
            model = get_targeted_model(self.base_model, self.mutators, self.sample)
            _logger.info(f'New model created: {self.sample}')
            submit_models(model)
            wait_models(model)
            if model.status == ModelStatus.Failed:
                return self.reset(), 0., False, {}
            rew = model.metric
            _logger.info(f'Model metric received as reward: {rew}')
            return obs, rew, True, {}
        else:

            return obs, 0., False, {}


class Preprocessor(nn.Module):
    def __init__(self, obs_space, hidden_dim=64, num_layers=1):
        super().__init__()
        self.action_dim = obs_space['action_history'].nvec[0]
        self.hidden_dim = hidden_dim
        # first token is [SOS]
        self.embedding = nn.Embedding(self.action_dim + 1, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, obs):
        seq = nn.functional.pad(obs['action_history'] + 1, (1, 1))  # pad the start token and end token
        # end token is used to avoid out-of-range of v_s_. Will not actually affect BP.
        seq = self.embedding(seq.long())
        feature, _ = self.rnn(seq)
        return feature[torch.arange(len(feature), device=feature.device), obs['cur_step'].long() + 1]


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
        mask = torch.arange(self.action_dim).expand(len(out), self.action_dim) >= obs['action_dim'].unsqueeze(1)
        out[mask.to(out.device)] = float('-inf')
        return nn.functional.softmax(out), kwargs.get('state', None)


class Critic(nn.Module):
    def __init__(self, preprocess):
        super().__init__()
        self.preprocess = preprocess
        self.linear = nn.Linear(self.preprocess.hidden_dim, 1)

    def forward(self, obs, **kwargs):
        obs = to_torch(obs, device=self.linear.weight.device)
        return self.linear(self.preprocess(obs)).squeeze(-1)
