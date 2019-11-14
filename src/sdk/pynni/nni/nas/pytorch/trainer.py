from abc import ABC, abstractmethod


class Trainer(ABC):

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def export(self):
        raise NotImplementedError
