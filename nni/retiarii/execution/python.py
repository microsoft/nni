from typing import Dict, Any, Type

import torch.nn as nn

from ..graph import Evaluator, Model
from ..integration_api import receive_trial_parameters
from ..utils import ContextStack
from .base import BaseExecutionEngine
from .utils import get_mutation_dict, mutation_dict_to_summary


class PythonGraphData:
    def __init__(self, class_: Type[nn.Module], init_parameters: Dict[str, Any],
                 mutation: Dict[str, Any], evaluator: Evaluator) -> None:
        self.class_ = class_
        self.init_parameters = init_parameters
        self.mutation = mutation
        self.evaluator = evaluator
        self.mutation_summary = mutation_dict_to_summary(mutation)

    def dump(self) -> dict:
        return {
            'class': self.class_,
            'init_parameters': self.init_parameters,
            'mutation': self.mutation,
            # engine needs to call dump here,
            # otherwise, evaluator will become binary
            # also, evaluator can be none in tests
            'evaluator': self.evaluator._dump() if self.evaluator is not None else None,
            'mutation_summary': self.mutation_summary
        }

    @staticmethod
    def load(data) -> 'PythonGraphData':
        return PythonGraphData(data['class'], data['init_parameters'], data['mutation'], Evaluator._load(data['evaluator']))


class PurePythonExecutionEngine(BaseExecutionEngine):
    """
    This is the execution engine that doesn't rely on Python-IR converter.

    We didn't explicitly state this independency for now. Front-end needs to decide which converter / no converter
    to use depending on the execution type. In the future, that logic may be moved into this execution engine.

    The execution engine needs to store the class path of base model, and init parameters to re-initialize the model
    with the mutation dict in the context, so that the mutable modules are created to be the fixed instance on the fly.
    """

    @classmethod
    def pack_model_data(cls, model: Model) -> Any:
        mutation = get_mutation_dict(model)
        graph_data = PythonGraphData(model.python_class, model.python_init_params, mutation, model.evaluator)
        return graph_data

    @classmethod
    def trial_execute_graph(cls) -> None:
        graph_data = PythonGraphData.load(receive_trial_parameters())

        def _model():
            return graph_data.class_(**graph_data.init_parameters)

        with ContextStack('fixed', graph_data.mutation):
            graph_data.evaluator._execute(_model)
