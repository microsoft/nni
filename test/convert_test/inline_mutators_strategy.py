import json
import logging
import random
import os

from nni.retiarii import Model, submit_models, wait_models
from nni.retiarii import get_base_model_ir, get_specified_mutators, get_trainer
from nni.retiarii import Sampler

_logger = logging.getLogger(__name__)

class RandomSampler(Sampler):
    def choice(self, candidates, mutator, model, index):
        return random.choice(candidates)

def _replace_layer_choice(node, cell_type):
    ...

def _replace_input_choice(node, cell_type):
    ...

def inline_mutators_startegy():
    """
    there should be an input argument to specify the type of weight sharing cell,
    e.g, enas cell, darts cell, ...
    """
    try:
        _logger.info('stargety start...')
        while True:
            model = get_base_model_ir()
            _logger.info('extract inline mutators...')
            for graph in model.graphs:
                for node in graph.nodes:
                    if node.operation.type == '__torch__.nni.retiarii.model_apis.inline_mutators.LayerChoice':
                        _replace_layer_choice(node, 'darts_cell')
                    elif node.operation.type == '__torch__.nni.retiarii.model_apis.inline_mutators.InputChoice':
                        _replace_input_choice(node, 'darts_cell')
            # apply regular mutators if users also specify some regular mutators
            #applied_mutators = get_specified_mutators()
            #_logger.info('mutators: {}'.format(applied_mutators))
            #random_sampler = RandomSampler()
            #for mutator in applied_mutators:
            #    _logger.info('mutate model...')
            #    mutator.bind_sampler(random_sampler)
            #    model = mutator.apply(model)

            # get and apply training approach
            _logger.info('apply training approach...')
            trainer = get_trainer()
            model.apply_trainer(trainer['modulename'], trainer['args'])

            # run models
            submit_models(model)
            wait_models(model)
            _logger.info('Strategy says:', model.metric)
    except Exception as e:
        _logger.error(logging.exception('message'))


if __name__ == '__main__':
    simple_startegy()
