from __future__ import annotations

import pytest

import nni
from nni.mutable import *
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.execution import SequentialExecutionEngine
from nni.nas.space import *
from nni.nas.strategy import GridSearch, Random, RegularizedEvolution, PolicyBasedRL

@nni.trace
class MyModelSpace(BaseModelSpace):
    def __init__(self, mutables: list[Mutable]):
        self.mutables = MutableList(mutables)

        if current_model() is not None:
            self.numbers = ensure_frozen(self.mutables)

    def call(self, x):
        return sum(n for n in self.numbers if n is not None) + x

    def leaf_mutables(self, is_leaf):
        yield from self.mutables.leaf_mutables(is_leaf)

    def check_contains(self, sample):
        for mutable in self.mutables:
            exc = mutable.check_contains(sample)
            if exc is not None:
                return exc
        return None

    def freeze(self, sample):
        with model_context(sample):
            return MyModelSpace(self.mutables)

def foo(model, a, optimal, minimal):
    nni.report_intermediate_result(a)
    nni.report_final_result((model.call(a) - minimal) / (optimal - minimal))

@pytest.fixture(params=['base', 'constraint', 'numerical'])
def named_model_space(request):
    if request.param == 'base':
        model = MyModelSpace([nni.choice('a', [1, 2, 3]), nni.choice('b', [4, 5, 6, 7])])
        evaluator = FunctionalEvaluator(foo, a=nni.choice('c', [0, 1]), optimal=11., minimal=5.)
    if request.param == 'constraint':
        a, b = nni.choice('a', [1, 2, 3]), nni.choice('b', [4, 5, 6, 7])
        model = MyModelSpace([a, b, ExpressionConstraint((a + b) % 2 == 1, label='odd')])
        evaluator = FunctionalEvaluator(foo, a=nni.choice('c', [0, 1]), optimal=10., minimal=5.)
    if request.param == 'numerical':
        model = MyModelSpace([nni.quniform('a', 1, 3, 0.75), nni.quniform('b', 4, 7, 1)])
        evaluator = FunctionalEvaluator(foo, a=1, optimal=11., minimal=6.)
    return request.param, SimplifiedModelSpace.from_model(model, evaluator)

@pytest.fixture
def engine():
    return SequentialExecutionEngine(max_model_count=30)

@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('dedup', [False, True])
def test_grid_search(named_model_space, engine, shuffle, dedup):
    name, model_space = named_model_space
    strategy = GridSearch(shuffle=shuffle, dedup=dedup)
    assert repr(strategy) == f'GridSearch(shuffle={shuffle}, dedup={dedup})'
    strategy(model_space, engine)
    if name == 'base':
        assert len(list(engine.list_models())) == 24
    elif name == 'constraint':
        assert len(list(engine.list_models())) == 12
    if name != 'numerical':
        samples = [model.sample for model in engine.list_models()]
        expected_samples = [model.sample for model in model_space.grid()]
        if shuffle:
            assert samples != expected_samples
        else:
            assert samples == expected_samples
    if shuffle and not dedup:
        assert next(strategy.list_models()).metric > 0.8
    else:
        assert next(strategy.list_models()).metric == 1.0  # optimal is always normalized to be 1.0

    with pytest.raises(RuntimeError, match='already been initialized'):
        strategy(model_space, engine)

    # Strategy resume.
    state_dict = strategy.state_dict()
    previous_submitted = len(list(engine.list_models()))
    if name == 'numerical':
        if shuffle or not dedup:
            assert state_dict['no_sample_found_counter'] == 0
        else:
            assert state_dict['no_sample_found_counter'] > 0
        if not shuffle:  # otherwise the first shuffle batch could not be exhausted.
            assert state_dict['granularity'] > 1
    else:
        assert state_dict['no_sample_found_counter'] == 0
        assert state_dict['granularity'] == 1
    assert 'random_state' in state_dict and isinstance(state_dict['random_state'], tuple)
    strategy2 = GridSearch(shuffle=shuffle, dedup=dedup)
    strategy2.load_state_dict(state_dict)

    strategy2(model_space, engine)
    if dedup:
        assert len(list(engine.list_models())) == previous_submitted

@pytest.mark.parametrize('dedup', [False, True])
def test_random(named_model_space, engine, dedup):
    name, model_space = named_model_space
    strategy = Random(dedup=dedup)
    assert repr(strategy) == f'Random(dedup={dedup})'
    strategy(model_space, engine)
    if dedup:
        if name != 'numerical':
            assert len(list(engine.list_models())) == len(list(model_space.grid()))
        assert next(strategy.list_models()).metric == 1.0

    state_dict = strategy.state_dict()
    previous_submitted = len(list(engine.list_models()))
    strategy2 = Random(dedup=dedup)
    strategy2.load_state_dict(state_dict)

    engine.max_model_count += 10
    strategy2(model_space, engine)
    if dedup:
        assert len(list(engine.list_models())) == previous_submitted
    else:
        assert len(list(engine.list_models())) == engine.max_model_count

# for failure rate testing
# @pytest.mark.parametrize('seed', list(range(50)))
@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize('crossover', [False, True])
def test_evolution(named_model_space, engine, crossover):
    # Failure rate is approximately 1/50 (2%) when distinct is not set.
    name, model_space = named_model_space
    strategy = RegularizedEvolution(population_size=10, sample_size=5, crossover=crossover, mutation_prob=0.3, dedup=False)
    assert repr(strategy) == f'RegularizedEvolution(population_size=10, sample_size=5, mutation_prob=0.3, crossover={crossover}, dedup=False)'
    strategy(model_space, engine)
    if name != 'constraint':
        assert next(strategy.list_models()).metric >= 0.7
    else:
        # constraint space is small.
        assert next(strategy.list_models()).metric == 1.0

    state_dict = strategy.state_dict()
    strategy2 = RegularizedEvolution(population_size=10, sample_size=5, crossover=crossover, mutation_prob=0.3, dedup=False)
    strategy2.load_state_dict(state_dict)

    strategy2(model_space, engine)
    assert len(list(engine.list_models())) == engine.max_model_count

@pytest.mark.parametrize('crossover', [False, True])
def test_evolution_dedup(named_model_space, engine, crossover):
    name, model_space = named_model_space
    strategy = RegularizedEvolution(population_size=10, sample_size=5, crossover=crossover, mutation_prob=0.2, dedup=True)
    strategy(model_space, engine)
    assert next(strategy.list_models()).metric == 1.0

    state_dict = strategy.state_dict()
    strategy2 = RegularizedEvolution(population_size=10, sample_size=5, crossover=crossover, mutation_prob=0.3, dedup=True)
    strategy2.load_state_dict(state_dict)

    strategy2(model_space, engine)
    assert len(list(engine.list_models())) < engine.max_model_count

# for failure rate testing
# @pytest.mark.parametrize('seed', list(range(50)))
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('reward_for_invalid', [None, -1.0])
def test_reinforcement_learning(named_model_space, engine, reward_for_invalid, caplog):
    # Failure rate for the base case is approximately 3/50 (6%).
    # Sometimes the first 20 samples are within an awkward range, making the policy hard to optimize.

    # import torch, numpy
    # torch.manual_seed(seed)
    # numpy.random.seed(seed)

    name, model_space = named_model_space
    strategy_kwargs = dict(
        samples_per_update=20,
        replay_buffer_size=200,
        policy_fn=naive_policy,
        update_kwargs={'batch_size': 50, 'repeat': 50, 'update_times': 5},
        reward_for_invalid=reward_for_invalid
    )
    strategy = PolicyBasedRL(**strategy_kwargs)
    assert repr(strategy) == f'PolicyBasedRL(samples_per_update=20, replay_buffer_size=200, reward_for_invalid={reward_for_invalid})'
    if name == 'numerical':
        with pytest.raises(ValueError, match='Categorical'):
            strategy(model_space, engine)
        return

    strategy(model_space, engine)

    # Failure rate for constraint + reward_for_invalid is approximately 9/50 (18%).
    # The odd condition implies mutual dependency between variables, unfriendly to the simple policy.
    if name != 'base' and reward_for_invalid is not None:
        assert next(strategy.list_models()).metric >= 0.8
    else:
        assert next(strategy.list_models()).metric == 1.0

    if name == 'constraint' and reward_for_invalid == -1:
        return  # FIXME: fails too often

    prev_models = list(engine.list_models())
    state_dict = strategy.state_dict()
    strategy2 = PolicyBasedRL(**strategy_kwargs)
    strategy2.load_state_dict(state_dict)
    assert 'not restored' in caplog.text
    engine.max_model_count += 10

    strategy2.initialize(model_space, engine)
    strategy2.load_state_dict(state_dict)

    strategy2.run()
    new_models = [m for m in engine.list_models() if m not in prev_models]
    assert len(new_models) == 10
    if name == 'base':
        assert max(m.metric for m in new_models) == 1.0


# A simple policy that is easier to optimize for this case.

import torch
from torch import nn
from tianshou.policy import A2CPolicy
from tianshou.data import to_torch

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.)

    def forward(self, obs, **kwargs):
        obs = to_torch(obs, device=self.linear.weight.device)
        steps_onehot = nn.functional.one_hot(obs['cur_step'].long(), self.input_dim).float()
        out = self.linear(steps_onehot)
        mask = torch.arange(self.output_dim).expand(len(out), self.output_dim) >= obs['action_dim'].unsqueeze(1)
        out_bias = torch.zeros_like(out)
        out_bias.masked_fill_(mask, float('-inf'))
        return nn.functional.softmax(out + out_bias, dim=-1), kwargs.get('state', None)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.)

    def forward(self, obs, **kwargs):
        obs = to_torch(obs, device=self.linear.weight.device)
        steps_onehot = nn.functional.one_hot(obs['cur_step'].long(), self.input_dim).float()
        return self.linear(steps_onehot)

def naive_policy(env):
    actor = ActorNetwork(env.observation_space['cur_step'].n, env.action_space.n)
    critic = CriticNetwork(env.observation_space['cur_step'].n)
    policy = A2CPolicy(actor, critic,
                       torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1.),
                       torch.distributions.Categorical, discount_factor=1.)
    return policy
