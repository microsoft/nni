from .base_optimizer import BaseOptimizer
from .logical_plan import LogicalPlan


class WeightSharingOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        pass

    def convert(self, logical_plan: LogicalPlan) -> None:
        pass
