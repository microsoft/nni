# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['ModelEventType', 'ModelEvent', 'FinalMetricEvent', 'IntermediateMetricEvent', 'TrainingEndEvent']

from enum import Enum
from typing import ClassVar
from dataclasses import dataclass

from nni.nas.space import ExecutableModelSpace, ModelStatus
from nni.typehint import TrialMetric


class ModelEventType(str, Enum):
    """Type of a model update event."""
    FinalMetric = 'final_metric'
    IntermediateMetric = 'intermediate_metric'
    TrainingEnd = 'training_end'


@dataclass
class ModelEvent:
    """Event of a model update."""
    event_type: ClassVar[ModelEventType]
    model: ExecutableModelSpace

    def __post_init__(self):
        self._canceled: bool = False
        self._default_canceled: bool = False

    def stop_propagation(self):
        """Stop propagation of this event to other un-notified listeners.

        This is similar to ``event.stopImmediatePropagation()`` in JavaScript.
        """
        self._canceled = True

    def prevent_default(self):
        """Prevent the default action of this event.

        The default action is invoked at the end of the event dispatch.
        It's usually defined by whoever dispatches the event.

        This is similar to ``event.preventDefault()`` in JavaScript.
        """
        self._default_canceled = True


@dataclass
class FinalMetricEvent(ModelEvent):
    """Event of a model update with final metric.

    Currently the metric is raw, and wasn't canonicalized.
    But it's subject to change in next iterations.
    """
    event_type: ClassVar[ModelEventType] = ModelEventType.FinalMetric
    metric: TrialMetric


@dataclass
class IntermediateMetricEvent(ModelEvent):
    """Event of a model update with intermediate metric."""
    event_type: ClassVar[ModelEventType] = ModelEventType.IntermediateMetric
    metric: TrialMetric


@dataclass
class TrainingEndEvent(ModelEvent):
    """Event of a model update with training end."""
    event_type: ClassVar[ModelEventType] = ModelEventType.TrainingEnd
    status: ModelStatus
