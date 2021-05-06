from typing import Dict, Any, List

from ..graph import Evaluator, Model
from ..integration_api import receive_trial_parameters
from ..utils import ContextStack, import_, get_importable_name
from .base import BaseExecutionEngine


class PythonGraphData:
    def __init__(self, model_class_name: str, mutation: Dict[str, Any], evaluator: Evaluator) -> None:
        self.model_class_name = model_class_name
        self.mutation = mutation
        self.evaluator = evaluator

    def dump(self) -> dict:
        return {
            'model_class_name': self.model_class_name,
            'mutation': self.mutation,
            'evaluator': self.evaluator
        }

    @staticmethod
    def load(data) -> 'PythonGraphData':
        return PythonGraphData(data['model_class_name'], data['mutation'], data['evaluator'])


class PurePythonExecutionEngine(BaseExecutionEngine):
    @classmethod
    def pack_model_data(cls, model: Model) -> Any:
        mutation = {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model.history}
        graph_data = PythonGraphData(get_importable_name(model.python_class, relocate_module=True),
                                     mutation, model.evaluator)
        return graph_data

    @classmethod
    def trial_execute_graph(cls) -> None:
        graph_data = PythonGraphData.load(receive_trial_parameters())
        with ContextStack('fixed', graph_data.mutation):
            model_cls = import_(graph_data.model_class_name)
            graph_data.evaluator._execute(model_cls)


def _unpack_if_only_one(ele: List[Any]):
    if len(ele) == 1:
        return ele[0]
    return ele
