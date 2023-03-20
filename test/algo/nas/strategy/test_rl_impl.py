from functools import partial

import numpy as np
import pytest
from tianshou.data import ReplayBuffer, Batch
from tianshou.policy import BasePolicy
from nni.mutable import Categorical, Numerical, CategoricalMultiple, MutableDict, MutableList, ExpressionConstraint
from nni.nas.strategy._rl_impl import TuningTrajectoryGenerator, default_policy_fn


def _has_upward_trend(arr):
    from scipy.stats import spearmanr
    corr = spearmanr(range(len(arr)), arr)
    return corr[0] > 0 and corr[1] < 0.1


def test_small_space():
    def eval_fn(v):
        return -(v ** 4 - 5 * v ** 2 - 3 * v)

    search_space = Categorical(range(-3, 4), label='x')
    best = max(map(eval_fn, search_space.grid()))

    replay_buffer = ReplayBuffer(20)
    generator = TuningTrajectoryGenerator(search_space, partial(default_policy_fn, hidden_dim=16, lr=1e-3))
    policy = generator.policy

    reward_curve = []
    for __ in range(50):
        rewards = []
        for _ in range(20):
            sample = generator.next_sample()
            sample_logits = generator.sample_logits
            reward = eval_fn(search_space.freeze(sample))
            rewards.append(reward)
            trajectory = generator.send_reward(reward)
            assert trajectory.act.shape == (1,)
            assert trajectory.done.shape == (1,)
            assert trajectory.info.is_empty()
            assert set(trajectory.obs.keys()) == {'action_history', 'cur_step', 'action_dim'}
            assert set(trajectory.obs_next.keys()) == {'action_history', 'cur_step', 'action_dim'}
            assert trajectory.obs.action_history.shape == (1, 1)
            assert trajectory.obs.cur_step.shape == (1,)
            assert trajectory.obs.action_dim.shape == (1,)
            replay_buffer.update(trajectory)

        policy.update(0, replay_buffer, batch_size=10, repeat=10)
        reward_curve.append(np.mean(rewards))

    assert _has_upward_trend(reward_curve)

    rewards = []
    for _ in range(3):
        sample = generator.next_sample()
        sample_logits = generator.sample_logits
        assert np.max(sample_logits['x']) > 0.9
        reward = eval_fn(search_space.freeze(sample))
        rewards.append(reward)
        if reward == best:
            break
    else:
        assert False, f'Cannot find the best sample, found: {rewards}'


def test_non_reuse_generator():
    def eval_fn(v):
        return -(v ** 4 - 5 * v ** 2 - 3 * v)

    search_space = Categorical(range(-3, 4), label='x')
    best = max(map(eval_fn, search_space.grid()))

    replay_buffer = ReplayBuffer(20)
    generator = TuningTrajectoryGenerator(search_space, partial(default_policy_fn, hidden_dim=16, lr=1e-3))
    policy = generator.policy

    reward_curve = []
    for __ in range(50):
        rewards = []
        generators = []
        for _ in range(20):
            generator = TuningTrajectoryGenerator(search_space, policy=policy)
            sample = generator.next_sample()
            generators.append(generator)
            reward = eval_fn(search_space.freeze(sample))
            rewards.append(reward)

        for generator, reward in zip(generators, rewards):
            trajectory = generator.send_reward(reward)
            replay_buffer.update(trajectory)

        policy.update(0, replay_buffer, batch_size=10, repeat=10)
        reward_curve.append(np.mean(rewards))

    assert _has_upward_trend(reward_curve)

    rewards = []
    for _ in range(3):
        sample = generator.next_sample()
        sample_logits = generator.sample_logits
        assert np.max(sample_logits['x']) > 0.9
        reward = eval_fn(search_space.freeze(sample))
        rewards.append(reward)
        if reward == best:
            break
    else:
        assert False, f'Cannot find the best sample, found: {rewards}'


def test_rl_synthetic_tuning():
    def eval_fn(sample):
        a, b, c = sample['a'], sample['b'], sample['c']
        return -(a - 1) * (a - 5) + b[0] - b[1] + sum(c)

    search_space = MutableDict({
        'a': Categorical([5, 3, 1], label='a'),
        'b': MutableList([
            Categorical([-1, 1, 3, 5], label='b1'),
            Categorical([1, 0], label='b2'),
        ]),
        'c': CategoricalMultiple([1, 2], label='c')
    })

    best = max(map(eval_fn, search_space.grid()))

    replay_buffer = ReplayBuffer(1000)
    generator = TuningTrajectoryGenerator(search_space, partial(default_policy_fn, hidden_dim=16, lr=1e-3))
    policy = generator.policy

    reward_curve = []
    for iter in range(100):
        rewards = []
        for _ in range(20):
            sample = generator.next_sample()
            sample_logits = generator.sample_logits
            reward = eval_fn(search_space.freeze(sample))
            rewards.append(reward)
            trajectory = generator.send_reward(reward)
            replay_buffer.update(trajectory)

        if iter > 10 and all(np.max(logits) > 0.9 for logits in sample_logits.values()):
            break

        policy.update(0, replay_buffer, batch_size=10, repeat=10)
        reward_curve.append(np.mean(rewards))
    else:
        assert False, f'Failed to converge.'

    assert _has_upward_trend(reward_curve)

    rewards = []
    for _ in range(5):
        sample = generator.next_sample()
        reward = eval_fn(search_space.freeze(sample))
        rewards.append(reward)
        if reward == best:
            break
    else:
        assert False, f'Cannot find the best sample, found: {rewards}'


def test_raises():
    search_space = Numerical(0, 1, label='a')
    with pytest.raises(ValueError):
        # TODO: improve message
        TuningTrajectoryGenerator(search_space)


class HiddenStatePolicy(BasePolicy):
    def __init__(self, env):
        super().__init__(env.observation_space, env.action_space)

    def forward(self, batch, state=None, **kwargs):
        batch_size = batch.obs.cur_step.shape[0]
        if state is None:
            state = np.zeros(batch_size)
        else:
            state = state + 2
        return Batch(
            act=np.zeros(batch_size, dtype=np.int64),
            state=state
        )

    def learn(self, batch, **kwargs):
        # dummy forward
        for minibatch in batch.split(2):
            self(minibatch)


def test_hidden_state_policy():
    search_space = CategoricalMultiple([1, 2], label='x')

    replay_buffer = ReplayBuffer(20)
    generator = TuningTrajectoryGenerator(search_space, HiddenStatePolicy)
    policy = generator.policy

    for _ in range(20):
        sample = generator.next_sample()
        assert not generator.sample_logits
        reward = sum(search_space.freeze(sample))
        trajectory = generator.send_reward(reward)
        assert np.all(trajectory.policy.hidden_state == np.array([0, 2]))
        assert np.all(trajectory.rew == np.array([0, 3]))
        assert np.all(trajectory.done == np.array([False, True]))
        replay_buffer.update(trajectory)

    policy.update(0, replay_buffer)


def test_categorical_multiple():
    search_space = MutableList([
        CategoricalMultiple([1, 2], label='x'),
        CategoricalMultiple([3, 4, 5], n_chosen=1, label='y')
    ])

    replay_buffer = ReplayBuffer(20)
    generator = TuningTrajectoryGenerator(search_space)
    policy = generator.policy

    for _ in range(20):
        sample = generator.next_sample()
        assert isinstance(sample['x'], list) and 0 <= len(sample['x']) <= 2
        assert isinstance(sample['y'], list) and len(sample['y']) == 1
        assert isinstance(generator.sample_logits['x'], list) and len(generator.sample_logits['x']) == 2
        assert isinstance(generator.sample_logits['y'], list) and len(generator.sample_logits['y']) == 3
        search_space.freeze(sample)
        trajectory = generator.send_reward(1.)
        replay_buffer.update(trajectory)

    policy.update(0, replay_buffer, batch_size=10, repeat=2)
