from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train(self, validate=True):
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        raise NotImplementedError

    @abstractmethod
    def export(self, file):
        raise NotImplementedError
