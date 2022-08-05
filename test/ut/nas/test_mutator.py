import json
from pathlib import Path

from nni.common.framework import get_default_framework, set_default_framework
from nni.retiarii import *

original_framework = get_default_framework()

max_pool = Operation.new('MaxPool2D', {'pool_size': 2})
avg_pool = Operation.new('AveragePooling2D', {'pool_size': 2})
global_pool = Operation.new('GlobalAveragePooling2D')


def setup_module(module):
    set_default_framework('tensorflow')


def teardown_module(module):
    set_default_framework(original_framework)


class DebugSampler(Sampler):
    def __init__(self):
        self.iteration = 0

    def choice(self, candidates, mutator, model, index):
        idx = (self.iteration + index) % len(candidates)
        return candidates[idx]

    def mutation_start(self, mutator, model):
        self.iteration += 1


class DebugMutator(Mutator):
    def mutate(self, model):
        ops = [max_pool, avg_pool, global_pool]

        pool1 = model.graphs['stem'].get_node_by_name('pool1')
        pool1.update_operation(self.choice(ops))

        pool2 = model.graphs['stem'].get_node_by_name('pool2')
        pool2.update_operation(self.choice(ops))


sampler = DebugSampler()
mutator = DebugMutator()
mutator.bind_sampler(sampler)


json_path = Path(__file__).parent / 'mnist-tensorflow.json'
ir = json.load(json_path.open())
model0 = Model._load(ir)


def test_dry_run():
    candidates, _ = mutator.dry_run(model0)
    assert len(candidates) == 2
    assert candidates[0] == [max_pool, avg_pool, global_pool]
    assert candidates[1] == [max_pool, avg_pool, global_pool]


def test_mutation():
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


def _get_pools(model):
    pool1 = model.graphs['stem'].get_node_by_name('pool1').operation
    pool2 = model.graphs['stem'].get_node_by_name('pool2').operation
    return pool1, pool2


if __name__ == '__main__':
    setup_module(None)
    test_dry_run()
    test_mutation()
    teardown_module(None)
