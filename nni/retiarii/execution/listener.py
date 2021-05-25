# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..graph import Model, ModelStatus
from .interface import MetricData, AbstractGraphListener


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
