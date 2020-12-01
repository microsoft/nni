import abc
from typing import List

class BaseStrategy(abc.ABC):
    @abc.abstractmethod
    def run(self, base_model: 'Model', applied_mutators: List['Mutator'], trainer: 'BaseTrainer') -> None:
        pass
