import json
from pathlib import Path

import pytest
from nni.common.framework import get_default_framework, set_default_framework
from nni.nas.space import StationaryMutator, Mutator, MutationSampler, GraphModelSpace, ModelStatus, MutatorSequence
from nni.nas.space.mutator import _RandomSampler
from nni.nas.space.graph_op import Operation


@pytest.fixture(autouse=True, scope='module')
def default_framework():
    original_framework = get_default_framework()
    set_default_framework('tensorflow')
    yield
    set_default_framework(original_framework)

@pytest.fixture(autouse=True)
def max_pool():
    yield Operation.new('MaxPool2D', {'pool_size': 2})

@pytest.fixture(autouse=True)
def avg_pool():
    yield Operation.new('AveragePooling2D', {'pool_size': 2})

@pytest.fixture(autouse=True)
def global_pool():
    yield Operation.new('GlobalAveragePooling2D')


class DebugSampler(MutationSampler):
    def __init__(self):
        self.iteration = 0

    def choice(self, candidates, mutator, model, index):
        idx = (self.iteration + index) % len(candidates)
        return candidates[idx]

    def mutation_start(self, mutator, model):
        self.iteration += 1


class DebugMutator(Mutator):
    def __init__(self, ops, label):
        super().__init__(label=label)
        self.ops = ops

    def mutate(self, model):
        pool1 = model.graphs['stem'].get_node_by_name('pool1')
        op = self.choice(self.ops)
        pool1.update_operation(op)

        pool2 = model.graphs['stem'].get_node_by_name('pool2')
        if op == self.ops[0]:
            pool2.update_operation(self.ops[0])
        else:
            pool2.update_operation(self.choice(self.ops))


class StationaryDebugMutator(StationaryMutator):
    def __init__(self, ops, label):
        super().__init__(label=label)
        self.ops = ops

    def mutate(self, model):
        pool1 = model.graphs['stem'].get_node_by_name('pool1')
        pool1.update_operation(self.choice(self.ops))

        pool2 = model.graphs['stem'].get_node_by_name('pool2')
        pool2.update_operation(self.choice(self.ops))


@pytest.fixture
def mutator(max_pool, avg_pool, global_pool):
    sampler = DebugSampler()
    mutator = StationaryDebugMutator(ops=[max_pool, avg_pool, global_pool], label='debug')
    mutator.bind_sampler(sampler)
    sampler.iteration = 0
    return mutator


@pytest.fixture
def mutator1(max_pool, avg_pool, global_pool):
    sampler = DebugSampler()
    mutator = DebugMutator(ops=[max_pool, avg_pool, global_pool], label='debug')
    mutator.bind_sampler(sampler)
    sampler.iteration = 0
    return mutator


@pytest.fixture
def model0():
    json_path = Path(__file__).parent / 'mnist_tensorflow.json'
    ir = json.load(json_path.open())
    return GraphModelSpace._load(_internal=True, **ir)


def test_dry_run(model0, mutator, max_pool, avg_pool, global_pool):
    assert model0.status == ModelStatus.Initialized
    candidates, model1 = mutator.dry_run(model0)
    assert model0.status == ModelStatus.Initialized
    assert model1.status == ModelStatus.Mutating
    assert len(candidates) == 2
    assert candidates['debug/0'].values == [max_pool, avg_pool, global_pool]
    assert candidates['debug/1'].values == [max_pool, avg_pool, global_pool]


def test_mutation(model0, mutator, max_pool, avg_pool, global_pool):
    model1 = mutator.apply(model0)
    assert _get_pools(model1) == (avg_pool, global_pool)

    model2 = mutator.apply(model1)
    assert _get_pools(model2) == (global_pool, max_pool)

    assert len(model2.history) == 2
    assert model2.history[0].from_ == model0
    assert model2.history[0].to == model1
    assert model2.history[1].from_ == model1
    assert model2.history[1].to == model2
    assert model2.history[0].mutator == mutator
    assert model2.history[1].mutator == mutator

    assert _get_pools(model0) == (max_pool, max_pool)
    assert _get_pools(model1) == (avg_pool, global_pool)


def test_mutator_sequence(model0, mutator, max_pool, avg_pool):
    mutators = MutatorSequence([mutator])
    with pytest.raises(AssertionError, match='bound to a model'):
        mutators.simplify()
    with mutators.bind_model(model0):
        assert list(mutators.simplify().keys()) == ['debug/0', 'debug/1']
    with mutators.bind_model(model0):
        model1 = mutators.freeze({'debug/0': avg_pool, 'debug/1': max_pool})
    assert model1.status == ModelStatus.Mutating
    assert len(model1.history) == 1
    assert _get_pools(model1) == (avg_pool, max_pool)


def test_simplify_and_random(model0, mutator, max_pool, avg_pool, global_pool):
    model0.mutators = MutatorSequence([mutator])
    assert list(model0.simplify().keys()) == ['debug/0', 'debug/1']
    mutator.sampler = None
    model1 = model0.random()
    assert model1.status == ModelStatus.Frozen
    assert list(model1.sample.keys()) == ['debug/0', 'debug/1']
    assert model1.sample['debug/0'] in [max_pool, avg_pool, global_pool]
    assert model1.sample['debug/1'] in [max_pool, avg_pool, global_pool]


def test_nonstationary_mutator(model0, mutator1, max_pool, avg_pool, global_pool):
    model = model0
    for _ in range(10):
        model = mutator1.apply(model)
        pools = _get_pools(model) 
        if pools[0] == max_pool:
            assert pools[1] == max_pool
        else:
            assert pools[0] in [avg_pool, global_pool]
            assert pools[1] in [max_pool, avg_pool, global_pool]


def test_nonstationary_mutator_simplify(model0, mutator1, max_pool, avg_pool, global_pool):
    model0.mutators = MutatorSequence([mutator1])
    assert model0.simplify() == {'debug': mutator1}
    mutator1.sampler = None
    model1 = model0.random()
    assert model1.status == ModelStatus.Frozen
    assert isinstance(model1.sample['debug'], _RandomSampler)
    pools = _get_pools(model1)
    assert pools[0] in [max_pool, avg_pool, global_pool]
    assert pools[1] in [max_pool, avg_pool, global_pool]


def _get_pools(model):
    pool1 = model.graphs['stem'].get_node_by_name('pool1').operation
    pool2 = model.graphs['stem'].get_node_by_name('pool2').operation
    return pool1, pool2
