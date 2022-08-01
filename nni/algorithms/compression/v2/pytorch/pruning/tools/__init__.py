# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base import (
    HookCollectorInfo,
    DataCollector,
    MetricsCalculator,
    SparsityAllocator,
    TaskGenerator
)
from .data_collector import (
    TargetDataCollector,
    EvaluatorBasedTargetDataCollector,
    EvaluatorBasedHookDataCollector
)
# TODO: remove in nni v3.0.
from .data_collector import (
    WeightDataCollector,
    WeightTrainerBasedDataCollector,
    SingleHookTrainerBasedDataCollector
)
from .metrics_calculator import (
    StraightMetricsCalculator,
    NormMetricsCalculator,
    HookDataNormMetricsCalculator,
    DistMetricsCalculator,
    APoZRankMetricsCalculator,
    MeanRankMetricsCalculator
)
from .sparsity_allocator import (
    NormalSparsityAllocator,
    BankSparsityAllocator,
    GlobalSparsityAllocator,
    DependencyAwareAllocator
)
from .task_generator import (
    AGPTaskGenerator,
    LinearTaskGenerator,
    LotteryTicketTaskGenerator,
    SimulatedAnnealingTaskGenerator
)
