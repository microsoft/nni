from ..graph import Model, ModelStatus
from .interface import MetricData, AbstractGraphListener


class DefaultListener(AbstractGraphListener):
    def __init__(self):
        self.resources: int = 0 # simply resource count

    def has_available_resource(self) -> bool:
        return self.resources > 0

    def on_metric(self, model: Model, metric: MetricData) -> None:
        model.metric = metric

    def on_intermediate_metric(self, model: Model, metric: MetricData) -> None:
        model.intermediate_metrics.append(metric)

    def on_training_end(self, model: Model, success: bool) -> None:
        if success:
            model.status = ModelStatus.Trained
        else:
            model.status = ModelStatus.Failed

    def on_resource_available(self, resources: int) -> None:
        self.resources += resources

    def on_resource_used(self, resources: int) -> None:
        self.resources -= resources
