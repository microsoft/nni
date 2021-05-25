import logging
from typing import Optional, Callable

from .base import BaseStrategy
from .utils import dry_run_for_search_space
from ..execution import query_available_resources

try:
    has_tianshou = True
    import torch
    from tianshou.data import AsyncCollector, Collector, VectorReplayBuffer
    from tianshou.env import SubprocVectorEnv
    from tianshou.policy import BasePolicy, PPOPolicy  # pylint: disable=unused-import
    from ._rl_impl import ModelEvaluationEnv, Preprocessor, Actor, Critic
except ImportError:
    has_tianshou = False


_logger = logging.getLogger(__name__)


class PolicyBasedRL(BaseStrategy):
    """
    Algorithm for policy-based reinforcement learning.
    This is a wrapper of algorithms provided in tianshou (PPO by default),
    and can be easily customized with other algorithms that inherit ``BasePolicy`` (e.g., REINFORCE [1]_).

    Note that RL algorithms are known to have issues on Windows and MacOS. They will be supported in future.

    Parameters
    ----------
    max_collect : int
        How many times collector runs to collect trials for RL. Default 100.
    trial_per_collect : int
        How many trials (trajectories) each time collector collects.
        After each collect, trainer will sample batch from replay buffer and do the update. Default: 20.
    policy_fn : function
        Takes ``ModelEvaluationEnv`` as input and return a policy. See ``_default_policy_fn`` for an example.
    asynchronous : bool
        If true, in each step, collector won't wait for all the envs to complete.
        This should generally not affect the result, but might affect the efficiency. Note that a slightly more trials
        than expected might be collected if this is enabled.
        If asynchronous is false, collector will wait for all parallel environments to complete in each step.
        See ``tianshou.data.AsyncCollector`` for more details.

    References
    ----------

    .. [1] Barret Zoph and Quoc V. Le, "Neural Architecture Search with Reinforcement Learning".
        https://arxiv.org/abs/1611.01578
    """

    def __init__(self, max_collect: int = 100, trial_per_collect = 20,
                 policy_fn: Optional[Callable[['ModelEvaluationEnv'], 'BasePolicy']] = None, asynchronous: bool = True):
        if not has_tianshou:
            raise ImportError('`tianshou` is required to run RL-based strategy. '
                              'Please use "pip install tianshou" to install it beforehand.')

        self.policy_fn = policy_fn or self._default_policy_fn
        self.max_collect = max_collect
        self.trial_per_collect = trial_per_collect
        self.asynchronous = asynchronous

    @staticmethod
    def _default_policy_fn(env):
        net = Preprocessor(env.observation_space)
        actor = Actor(env.action_space, net)
        critic = Critic(net)
        optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=1e-4)
        return PPOPolicy(actor, critic, optim, torch.distributions.Categorical,
                         discount_factor=1., action_space=env.action_space)

    def run(self, base_model, applied_mutators):
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        concurrency = query_available_resources()

        env_fn = lambda: ModelEvaluationEnv(base_model, applied_mutators, search_space)
        policy = self.policy_fn(env_fn())

        if self.asynchronous:
            # wait for half of the env complete in each step
            env = SubprocVectorEnv([env_fn for _ in range(concurrency)], wait_num=int(concurrency * 0.5))
            collector = AsyncCollector(policy, env, VectorReplayBuffer(20000, len(env)))
        else:
            env = SubprocVectorEnv([env_fn for _ in range(concurrency)])
            collector = Collector(policy, env, VectorReplayBuffer(20000, len(env)))

        for cur_collect in range(1, self.max_collect + 1):
            _logger.info('Collect [%d] Running...', cur_collect)
            result = collector.collect(n_episode=self.trial_per_collect)
            _logger.info('Collect [%d] Result: %s', cur_collect, str(result))
            policy.update(0, collector.buffer, batch_size=64, repeat=5)
