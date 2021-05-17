import logging
from nni.retiarii.execution.api import query_available_resources
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn

from .base import BaseStrategy
from .utils import dry_run_for_search_space, get_targeted_model
from ..execution import submit_models, wait_models

try:
    has_tianshou = True
    import gym
    import tianshou
    from gym import spaces
    from tianshou.data import to_torch, Collector, VectorReplayBuffer
    from tianshou.env import SubprocVectorEnv
    from tianshou.policy import BasePolicy, PPOPolicy
    from tianshou.trainer import onpolicy_trainer
except ImportError:
    has_tianshou = False


_logger = logging.getLogger(__name__)


class PolicyBasedRL(BaseStrategy):

    def __init__(self, max_collect: int = 100, episode_per_collect = 20,
                 policy_fn: Optional[Callable[[], BasePolicy]] = None):
        if not has_tianshou:
            raise ImportError('`tianshou` is required to run RL-based strategy. '
                              'Please use "pip install tianshou" to install it beforehand.')

        self.policy_fn = policy_fn or self._default_policy_fn
        self.max_collect = max_collect
        self.episode_per_collect = episode_per_collect

    @staticmethod
    def _default_policy_fn(env):
        net = PolicyBasedRL.Preprocessor(env.observation_space)
        actor = PolicyBasedRL.Actor(env.action_space, net)
        critic = PolicyBasedRL.Critic(net)
        optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=1e-4)
        return PPOPolicy(actor, critic, optim, torch.distributions.Categorical,
                         discount_factor=1., action_space=env.action_space)

    def run(self, base_model, applied_mutators):
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        concurrency = query_available_resources()

        env = SubprocVectorEnv([lambda: PolicyBasedRL.ModelEvaluationEnv(base_model, applied_mutators, search_space)
                                for _ in range(concurrency)])
        policy = self.policy_fn(env)
        collector = Collector(policy, env, VectorReplayBuffer(20000, len(env)))

        for cur_collect in range(1, self.max_collect + 1):
            result = collector.collect(n_episode=self.episode_per_collect)
            policy.update(0, collector.buffer, batch_size=64, repeat=5)

    # Environment for RL

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
                'action_dim': len(self.search_space[self.ss_keys[self.cur_step]])
            }
            if self.cur_step == self.num_steps:
                model = get_targeted_model(self.base_model, self.mutators, self.sample)
                submit_models(model)
                wait_models(model)
                rew = model.metric
                return obs, rew, True, {}
            else:
                return obs, 0., False, {}

    # Policy network for RL

    class Preprocessor(nn.Module):
        def __init__(self, obs_space, hidden_dim=64, num_layers=1):
            super().__init__()
            self.action_dim = obs_space['action_dim'].high
            self.hidden_dim = hidden_dim
            self.embedding = nn.Embedding(self.action_dim, hidden_dim)
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        def forward(self, obs):
            seq = self.embedding(obs['action_history'])
            feature, _ = self.net(seq)
            return feature[torch.arange(len(feature), device=feature.device), obs['cur_step'].long()]

    class Actor(nn.Module):
        def __init__(self, action_space, preprocess):
            super().__init__()
            self.preprocess = preprocess
            self.action_dim = action_space.high + 1
            self.linear = nn.Linear(self.preprocess.hidden_dim, self.action_dim)

        def forward(self, obs):
            obs = to_torch(obs, device=self.linear.device)
            out = self.linear(self.preprocess(obs))
            out[torch.arange(len(out), device=out.device), obs['action_dim']:] = float('-inf')
            return nn.functional.softmax(out)

    class Critic(nn.Module):
        def __init__(self, preprocess):
            super().__init__()
            self.preprocess = preprocess
            self.linear = nn.Linear(self.preprocess.hidden_dim, 1)

        def forward(self, obs):
            obs = to_torch(obs, device=self.linear.device)
            out = self.linear(self.extractor(obs)).squeeze(-1)

