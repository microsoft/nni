from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        raise NotImplementedError

    @abstractmethod
    def train_and_validate(self):
        raise NotImplementedError
