#import nni

from ._experiment import ExperimentConfig
from .experiment import create_experiment
from .execution import submit_graph, submit_graphs, train_graph, train_graphs, wait_graph, wait_graphs
from .mutators.mutator import Mutator
from ._trainer import Trainer

__all__ = [
    'experiment',

    'Mutator',

    'submit_graph',
    'submit_graphs',
    'train_graph',
    'train_graphs',
    'wait_graph',
    'wait_graphs',
    'Trainer',
    'report_final_result',
    'report_intermediate_result',
]


ExperimentConfig = ExperimentConfig()

# nni.report_xxx_result() will not be imported in dispatcher process, so don't use assigning

# def report_final_result(metric: 'Any') -> None:
#     nni.report_final_result(metric)

# def report_intermediate_result(metric: 'Any') -> None:
#     nni.report_intermediate_result
