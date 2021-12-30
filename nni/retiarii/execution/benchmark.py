import os
import random
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Iterable

from ..graph import Model
from ..integration_api import receive_trial_parameters
from .base import BaseExecutionEngine
from .utils import get_mutation_dict


class BenchmarkGraphData:

    SUPPORTED_BENCHMARK_LIST = [
        'nasbench101',
        'nasbench201-cifar10',
        'nasbench201-cifar100',
        'nasbench201-imagenet16',
        'nds-cifar10',
        'nds-imagenet',
        'nlp'
    ]

    def __init__(self, mutation: Dict[str, Any], benchmark: str,
                 metric_name: Optional[str] = None,
                 db_path: Optional[str] = None) -> None:
        self.mutation = mutation        # mutation dict. e.g., {'layer1': 'conv3x3', ...}
        self.benchmark = benchmark      # e.g., nasbench101, nasbench201, ...
        self.db_path = db_path          # path to directory of database

    def dump(self) -> dict:
        from nni.nas.benchmarks.constants import DATABASE_DIR
        return {
            'mutation': self.mutation,
            'benchmark': self.benchmark,
            'db_path': self.db_path or DATABASE_DIR  # database path need to be passed from manager to worker
        }

    @staticmethod
    def load(data) -> 'BenchmarkGraphData':
        return BenchmarkGraphData(data['mutation'], data['benchmark'], data['metric_name'], data['db_path'])


class BenchmarkExecutionEngine(BaseExecutionEngine):
    """
    Execution engine that does not actually run any trial, but query the database for results.

    The database query is done on the trial end to make sure intermediate metrics are available.
    It will also support an accelerated mode that returns metric immediately without even running into NNI manager
    (not implemented yet).
    """

    def __init__(self, benchmark: Union[str, Callable[[BenchmarkGraphData], Tuple[float, List[float]]]], acceleration: bool = False):
        super().__init__()
        assert benchmark in BenchmarkGraphData.SUPPORTED_BENCHMARK_LIST, \
            f'{benchmark} is not one of the supported benchmarks: {BenchmarkGraphData.SUPPORTED_BENCHMARK_LIST}'
        self.benchmark = benchmark
        self.acceleration = acceleration

    def pack_model_data(self, model: Model) -> Any:
        # called when a new model is submitted to backend.
        # convert a Model into a data that is acceptable by trial end.
        mutation = get_mutation_dict(model)
        graph_data = BenchmarkGraphData(mutation, self.benchmark)

        return graph_data

    @classmethod
    def trial_execute_graph(cls) -> None:
        graph_data = BenchmarkGraphData.load(receive_trial_parameters())
        os.environ['NASBENCHMARK_DIR'] = graph_data.db_path
        final, intermediates = cls.query_in_benchmark(graph_data)

        import nni
        for i in intermediates:
            nni.report_intermediate_result(i)
        nni.report_final_result(final)

    @staticmethod
    def query_in_benchmark(graph_data: BenchmarkGraphData) -> Tuple[float, List[float]]:
        if not isinstance(graph_data.benchmark, str):
            return graph_data.benchmark(graph_data)

        # built-in benchmarks with default query setting
        if graph_data.benchmark == 'nasbench101':
            from nni.nas.benchmarks.nasbench101 import query_nb101_trial_stats
            arch = None
            for t in graph_data.mutation.values():
                if isinstance(t, dict):
                    arch = t
            if arch is None:
                raise ValueError(f'Cannot identify architecture from mutation dict: {graph_data.mutation}')
            print(arch)
            return _convert_to_final_and_intermediates(
                query_nb101_trial_stats(arch, 108, include_intermediates=True),
                'valid_acc'
            )
        elif graph_data.benchmark.startswith('nasbench201'):
            from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats
            dataset = graph_data.benchmark.split('-')[-1]
            return _convert_to_final_and_intermediates(
                query_nb201_trial_stats(_flatten_architecture(graph_data.mutation), 200, dataset, include_intermediates=True),
                'valid_acc',
            )
        elif graph_data.benchmark.startswith('nds'):
            # FIXME: not tested yet
            from nni.nas.benchmarks.nds import query_nds_trial_stats
            dataset = graph_data.benchmark.split('-')[-1]
            return _convert_to_final_and_intermediates(
                query_nds_trial_stats(None, None, None, None, _flatten_architecture(graph_data.mutation),
                                      dataset, include_intermediates=True),
                'valid_acc'
            )
        elif graph_data.benchmark.startswith('nlp'):
            # FIXME: not tested yet
            from nni.nas.benchmarks.nlp import query_nlp_trial_stats
            # TODO: I'm not sure of the availble datasets in this benchmark. and the docs are missing.
            return _convert_to_final_and_intermediates(
                query_nlp_trial_stats(_flatten_architecture(graph_data.mutation), 'ptb', include_intermediates=True),
                'valid_acc'
            )
        else:
            raise ValueError(f'{graph_data.benchmark} is not a supported benchmark.')


def _flatten_architecture(mutation: Dict[str, Any], benchmark: Optional[str] = None):
    # STRONG ASSUMPTION HERE!
    # This assumes that the benchmarked search space is a one-level search space.
    # This means that it is either ONE cell or ONE network.
    # Two cell search space like NDS is not supported yet for now.
    # Some benchmark even needs special handling to pop out invalid keys. I don't think this is a good design.

    # support double underscore to be compatible with naming convention in base engine
    ret = {k.split('/')[-1].split('__')[-1]: v for k, v in mutation.items()}
    if benchmark == 'nasbench101':
        ret = {k: v for k, v in ret.items() if k.startswith('op') or k.startswith('input')}
        ret = {k: v if k.startswith('op') or isinstance(v, list) else [v] for k, v in ret.items()}
    return ret


def _convert_to_final_and_intermediates(benchmark_result: Iterable[Any], metric_name: str) -> Tuple[float, List[float]]:
    # convert benchmark results from database to
    # final result (float) and intermediate results (list of floats)
    benchmark_result = list(benchmark_result)
    assert len(benchmark_result) > 0, 'Invalid query. Results from benchmark is empty.'
    if len(benchmark_result) > 1:
        benchmark_result = random.choice(benchmark_result)
    else:
        benchmark_result = benchmark_result[0]
    return benchmark_result[metric_name], [i[metric_name] for i in benchmark_result['intermediates'] if i[metric_name] is not None]
