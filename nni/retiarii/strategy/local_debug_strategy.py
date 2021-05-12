# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import random
import string

from .. import Sampler, codegen, utils
from ..execution.base import BaseGraphData
from .base import BaseStrategy

_logger = logging.getLogger(__name__)

class ChooseFirstSampler(Sampler):
    def choice(self, candidates, mutator, model, index):
        return candidates[0]

class _LocalDebugStrategy(BaseStrategy):
    """
    This class is supposed to be used internally, for debugging trial mutation
    """

    def run_one_model(self, model):
        graph_data = BaseGraphData(codegen.model_to_pytorch_script(model), model.evaluator)
        random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        file_name = f'_generated_model/{random_str}.py'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as f:
            f.write(graph_data.model_script)
        model_cls = utils.import_(f'_generated_model.{random_str}._model')
        graph_data.evaluator._execute(model_cls)
        os.remove(file_name)

    def run(self, base_model, applied_mutators):
        _logger.info('local debug strategy has been started.')
        model = base_model
        _logger.debug('New model created. Applied mutators: %s', str(applied_mutators))
        choose_first_sampler = ChooseFirstSampler()
        for mutator in applied_mutators:
            mutator.bind_sampler(choose_first_sampler)
            model = mutator.apply(model)
        # directly run models
        self.run_one_model(model)
