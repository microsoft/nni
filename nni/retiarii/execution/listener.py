from typing import *

from ..graph import *
from .interface import *


class DefaultListener(AbstractGraphListener):
    def __init__(self):
        self.resources: List[WorkerInfo] = []

    def on_metric(self, model: Model, metric: MetricData) -> None:
        model.metric = metric
            
    def on_intermediate_metric(self, model: Model, metric: MetricData) -> None:
        model.intermediate_metrics.append(metric)

    def on_training_end(self, model: Model, success: bool) -> None:
        if success:
            model.status = ModelStatus.Trained
        else:
            model.status = ModelStatus.Failed

    def on_resource_available(self, resources: List[WorkerInfo]) -> None:
        self.resources += resources

    def on_resource_used(self, resources: List[WorkerInfo]) -> None:
        self.resources = [r for r in self.resources if r not in resources]
