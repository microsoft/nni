from abc import ABC, abstractmethod


class Trainer(ABC):

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError
