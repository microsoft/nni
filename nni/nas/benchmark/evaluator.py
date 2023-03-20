# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['BenchmarkEvaluator', 'NasBench101Benchmark', 'NasBench201Benchmark']

import itertools
import logging
import random
import warnings
from typing import Any, cast

import nni
from nni.mutable import Sample, Mutable, LabeledMutable, Categorical, CategoricalMultiple, label_scope
from nni.nas.evaluator import Evaluator
from nni.nas.space import ExecutableModelSpace

from .space import SlimBenchmarkSpace

_logger = logging.getLogger(__name__)


def _report_intermediates_and_final(query_result: list[Any], metric: str, query: str, scale: float = 1.) -> tuple[float, list[float]]:
    """Convert benchmark results from database to results reported to NNI.

    Utility function for :meth:`BenchmarkEvaluator.evaluate`.
    """
    if not query_result:
        raise ValueError('Invalid query. Results from benchmark is empty: ' + query)
    if len(query_result) > 1:
        query_result = random.choice(query_result)
    else:
        query_result = query_result[0]
    query_dict = cast(dict, query_result)
    for i in query_dict.get('intermediates', []):
        if i[metric] is not None:
            nni.report_intermediate_result(i[metric] * scale)
    nni.report_final_result(query_dict[metric] * scale)
    return query_dict[metric]


def _common_label_scope(labels: list[str]) -> str:
    """Find the longest prefix of all labels.

    The prefix must ends with ``/``.
    """
    if not labels:
        return ''
    for i in range(len(labels[0]) - 1, -1, -1):
        if labels[0][i] == '/' and all(s.startswith(labels[0][:i + 1]) for s in labels):
            return labels[0][:i + 1]
    return ''


def _strip_common_label_scope(sample: Sample) -> Sample:
    """Strip the common label scope from the sample."""
    scope_name = _common_label_scope(list(sample))
    if not scope_name:
        return sample
    return {k[len(scope_name):]: v for k, v in sample.items()}


def _sorted_dict(sample: Sample) -> Sample:
    """Sort the keys of a dict."""
    return dict(sorted(sample.items()))


class BenchmarkEvaluator(Evaluator):
    """A special kind of evaluator that does not run real training, but queries a database."""

    @classmethod
    def default_space(cls) -> SlimBenchmarkSpace:
        """Return the default search space benchmarked by this evaluator.

        Subclass should override this.
        """
        raise NotImplementedError()

    def validate_space(self, space: Mutable) -> dict[str, LabeledMutable]:
        """Validate the search space. Raise exception if invalid. Returns the validated space.

        By default, it will cross-check with the :meth:`default_space`, and return the default space.
        Differences in common scope names will be ignored.

        I think the default implementation should work for most cases.
        But subclass can still override this method for looser or tighter validation.
        """
        current_space = space.simplify()

        scope_name = _common_label_scope(list(current_space))
        if not scope_name:
            default_space = self.default_space()
        else:
            with label_scope(scope_name.rstrip('/')):
                default_space = self.default_space()

        if SlimBenchmarkSpace(current_space) != default_space:
            raise ValueError(f'Expect space to be {default_space}, got {current_space}')

        return current_space

    def evaluate(self, sample: Sample) -> Any:
        """:meth:`evaluate` receives a sample and returns a float score.
        It also reports intermediate and final results through NNI trial API.

        Necessary format conversion and database query should be done in this method.

        It is the main interface of this class. Subclass should override this.
        """
        raise NotImplementedError()

    def _execute(self, model: ExecutableModelSpace) -> Any:
        """Execute the model with the sample."""

        from .space import BenchmarkModelSpace
        if not isinstance(model, BenchmarkModelSpace):
            warnings.warn('It would be better to use BenchmarkModelSpace for benchmarking to avoid '
                          'unnecessary overhead and silent mistakes.')
        if model.sample is None:
            raise ValueError('Model can not be evaluted because it has not been sampled yet.')

        return self.evaluate(model.sample)


class NasBench101Benchmark(BenchmarkEvaluator):
    """Benchmark evaluator for NAS-Bench-101.

    Parameters
    ----------
    num_epochs
        Queried ``num_epochs``.
    metric
        Queried metric.
    include_intermediates
        Whether to report intermediate results.

    See Also
    --------
    nni.nas.benchmark.nasbench101.query_nb101_trial_stats
    nni.nas.benchmark.nasbench101.Nb101TrialConfig
    """

    def __init__(self, num_epochs: int = 108, metric: str = 'valid_acc', include_intermediates: bool = False) -> None:
        super().__init__()
        self.metric = metric
        self.include_intermediates = include_intermediates
        self.num_epochs = num_epochs

    @classmethod
    def default_space(cls) -> SlimBenchmarkSpace:
        from nni.nas.hub.pytorch.modules.nasbench101 import NasBench101CellConstraint, NasBench101Cell
        op_candidates = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']

        # For readability, expand it here.
        num_nodes = NasBench101Cell._num_nodes_discrete(7)
        # num_nodes = Categorical([2, 3, 4, 5, 6, 7], label='num_nodes')
        ops = [
            Categorical(op_candidates, label='op1'),
            Categorical(op_candidates, label='op2'),
            Categorical(op_candidates, label='op3'),
            Categorical(op_candidates, label='op4'),
            Categorical(op_candidates, label='op5')
        ]
        inputs = [
            CategoricalMultiple([0], n_chosen=None, label='input1'),
            CategoricalMultiple([0, 1], n_chosen=None, label='input2'),
            CategoricalMultiple([0, 1, 2], n_chosen=None, label='input3'),
            CategoricalMultiple([0, 1, 2, 3], n_chosen=None, label='input4'),
            CategoricalMultiple([0, 1, 2, 3, 4], n_chosen=None, label='input5'),
            CategoricalMultiple([0, 1, 2, 3, 4, 5], n_chosen=None, label='input6')
        ]
        constraint = NasBench101CellConstraint(9, num_nodes, ops, inputs)

        return SlimBenchmarkSpace({
            mutable.label: mutable for mutable in itertools.chain([num_nodes], ops, inputs, [constraint])
        })

    def evaluate(self, sample: Sample) -> Any:
        sample = _sorted_dict(_strip_common_label_scope(sample))
        _logger.debug('NasBench101 sample submitted to query: %s', sample)

        from nni.nas.benchmark.nasbench101 import query_nb101_trial_stats
        query = query_nb101_trial_stats(sample['final'], self.num_epochs, include_intermediates=self.include_intermediates)
        return _report_intermediates_and_final(
            list(query), self.metric, str(sample), .01
        )


class NasBench201Benchmark(BenchmarkEvaluator):
    """Benchmark evaluator for NAS-Bench-201.

    Parameters
    ----------
    num_epochs
        Queried ``num_epochs``.
    dataset
        Queried ``dataset``.
    metric
        Queried metric.
    include_intermediates
        Whether to report intermediate results.

    See Also
    --------
    nni.nas.benchmark.nasbench201.query_nb201_trial_stats
    nni.nas.benchmark.nasbench201.Nb201TrialConfig
    """

    def __init__(self, num_epochs: int = 200, dataset: str = 'cifar100', metric: str = 'valid_acc',
                 include_intermediates: bool = False) -> None:
        super().__init__()
        self.metric = metric
        self.dataset = dataset
        self.include_intermediates = include_intermediates
        self.num_epochs = num_epochs

    @classmethod
    def default_space(cls) -> SlimBenchmarkSpace:
        operations = ['none', 'skip_connect', 'conv_1x1', 'conv_3x3', 'avg_pool_3x3']
        ops = [
            Categorical(operations, label='0_1'),
            Categorical(operations, label='0_2'),
            Categorical(operations, label='1_2'),
            Categorical(operations, label='0_3'),
            Categorical(operations, label='1_3'),
            Categorical(operations, label='2_3')
        ]
        return SlimBenchmarkSpace({op.label: op for op in ops})

    def evaluate(self, sample: Sample) -> Any:
        sample = _sorted_dict(_strip_common_label_scope(sample))
        _logger.debug('NasBench201 sample submitted to query: %s', sample)

        from nni.nas.benchmark.nasbench201 import query_nb201_trial_stats
        query = query_nb201_trial_stats(sample, self.num_epochs, self.dataset, include_intermediates=self.include_intermediates)
        return _report_intermediates_and_final(
            list(query), self.metric, str(sample), .01
        )
