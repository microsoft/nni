from .logical_plan import LogicalPlan
from .base_optimizer import BaseOptimizer


class BypassOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        pass

    def convert(self, logical_plan: LogicalPlan) -> None:
        pass