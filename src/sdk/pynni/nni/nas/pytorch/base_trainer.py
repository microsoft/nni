# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train(self):
        """
        Override the method to train.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        """
        Override the method to validate.
        """
        raise NotImplementedError

    @abstractmethod
    def export(self, file):
        """
        Override the method to export to file.

        Parameters
        ----------
        file : str
            File path to export to.
        """
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self):
        """
        Override to dump a checkpoint.
        """
        raise NotImplementedError
