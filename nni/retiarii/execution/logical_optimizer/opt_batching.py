from .base_optimizer import BaseOptimizer
from .logical_plan import LogicalPlan


class BatchingOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        pass

    def convert(self, logical_plan: LogicalPlan) -> None:
        pass
