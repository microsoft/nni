# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        raise NotImplementedError

    @abstractmethod
    def export(self, file):
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self):
        raise NotImplementedError
