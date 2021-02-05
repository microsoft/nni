from abc import ABC

from .logical_plan import LogicalPlan


class AbstractOptimizer(ABC):
    def __init__(self) -> None:
        pass

    def convert(self, logical_plan: LogicalPlan) -> None:
        raise NotImplementedError
