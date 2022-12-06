import json
from pathlib import Path

import pytest
from nni.common.framework import get_default_framework, set_default_framework
from nni.nas.space import StationaryMutator, MutationSampler, GraphModelSpace, ModelStatus, MutatorSequence
from nni.nas.space.graph_op import Operation

original_framework = get_default_framework()

max_pool = Operation.new('MaxPool2D', {'pool_size': 2})
avg_pool = Operation.new('AveragePooling2D', {'pool_size': 2})
global_pool = Operation.new('GlobalAveragePooling2D')


def setup_module(module):
    set_default_framework('tensorflow')


def teardown_module(module):
    set_default_framework(original_framework)


class DebugSampler(MutationSampler):
    def __init__(self):
        self.iteration = 0

    def choice(self, candidates, mutator, model, index):
        idx = (self.iteration + index) % len(candidates)
        return candidates[idx]

    def mutation_start(self, mutator, model):
        self.iteration += 1


class DebugMutator(StationaryMutator):
    def mutate(self, model):
        ops = [max_pool, avg_pool, global_pool]

        pool1 = model.graphs['stem'].get_node_by_name('pool1')
        pool1.update_operation(self.choice(ops))

        pool2 = model.graphs['stem'].get_node_by_name('pool2')
        pool2.update_operation(self.choice(ops))


sampler = DebugSampler()
mutator = DebugMutator(label='debug')
mutator.bind_sampler(sampler)


json_path = Path(__file__).parent / 'mnist_tensorflow.json'
ir = json.load(json_path.open())
model0 = GraphModelSpace._load(_internal=True, **ir)


def test_dry_run():
    assert model0.status == ModelStatus.Initialized
    candidates, model1 = mutator.dry_run(model0)
    assert model0.status == ModelStatus.Initialized
    assert model1.status == ModelStatus.Mutating
    assert len(candidates) == 2
    assert candidates['debug/0'].values == [max_pool, avg_pool, global_pool]
    assert candidates['debug/1'].values == [max_pool, avg_pool, global_pool]


def test_mutation():
    sampler.iteration = 0

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


def test_mutator_sequence():
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


def _get_pools(model):
    pool1 = model.graphs['stem'].get_node_by_name('pool1').operation
    pool2 = model.graphs['stem'].get_node_by_name('pool2').operation
    return pool1, pool2
