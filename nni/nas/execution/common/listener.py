# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = ['DefaultListener']

from .graph import Model, ModelStatus, MetricData
from .engine import AbstractGraphListener


class DefaultListener(AbstractGraphListener):

    def on_metric(self, model: Model, metric: MetricData) -> None:
        model.metric = metric

    def on_intermediate_metric(self, model: Model, metric: MetricData) -> None:
        model.intermediate_metrics.append(metric)

    def on_training_end(self, model: Model, success: bool) -> None:
        if success:
            model.status = ModelStatus.Trained
        else:
            model.status = ModelStatus.Failed
