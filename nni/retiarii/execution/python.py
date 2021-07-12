from typing import Dict, Any, List

from ..graph import Evaluator, Model
from ..integration_api import receive_trial_parameters
from ..utils import ContextStack, import_, get_importable_name
from .base import BaseExecutionEngine


class PythonGraphData:
    def __init__(self, class_name: str, init_parameters: Dict[str, Any],
                 mutation: Dict[str, Any], evaluator: Evaluator) -> None:
        self.class_name = class_name
        self.init_parameters = init_parameters
        self.mutation = mutation
        self.evaluator = evaluator

    def dump(self) -> dict:
        return {
            'class_name': self.class_name,
            'init_parameters': self.init_parameters,
            'mutation': self.mutation,
            'evaluator': self.evaluator
        }

    @staticmethod
    def load(data) -> 'PythonGraphData':
        return PythonGraphData(data['class_name'], data['init_parameters'], data['mutation'], data['evaluator'])


class PurePythonExecutionEngine(BaseExecutionEngine):
    @classmethod
    def pack_model_data(cls, model: Model) -> Any:
        mutation = get_mutation_dict(model)
        graph_data = PythonGraphData(get_importable_name(model.python_class, relocate_module=True),
                                     model.python_init_params, mutation, model.evaluator)
        return graph_data

    @classmethod
    def trial_execute_graph(cls) -> None:
        graph_data = PythonGraphData.load(receive_trial_parameters())

        class _model(import_(graph_data.class_name)):
            def __init__(self):
                super().__init__(**graph_data.init_parameters)

        with ContextStack('fixed', graph_data.mutation):
            graph_data.evaluator._execute(_model)


def _unpack_if_only_one(ele: List[Any]):
    if len(ele) == 1:
        return ele[0]
    return ele


def get_mutation_dict(model: Model):
    return {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model.history}
