# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Optional, Callable, Iterable

from torch.nn import Module
from torch.optim import Optimizer


class BaseAutoCompressionEngine(ABC):
    @classmethod
    @abstractmethod
    def trial_execute_compress(cls):
        """
        Execute the compressing trial.
        """
        pass


class AbstractAutoCompressionModule(ABC):
    """
    The abstract container that user need to implement.
    """
    @classmethod
    @abstractmethod
    def model(cls) -> Module:
        """
        Returns
        -------
        torch.nn.Module
            Model to be compress.
        """
        pass

    @classmethod
    @abstractmethod
    def evaluator(cls) -> Callable[[Module], float]:
        """
        Returns
        -------
        function
            The function used to evaluate the compressed model, return a scalar.
        """
        pass

    @classmethod
    @abstractmethod
    def optimizer_factory(cls) -> Optional[Callable[[Iterable], Optimizer]]:
        """
        Returns
        -------
        Optional[Callable[[Iterable], Optimizer]]
            Optimizer factory function. Input is a iterable value, i.e. `model.parameters()`.
            Output is the `torch.optim.Optimizer` instance.
        """
        pass

    @classmethod
    @abstractmethod
    def criterion(cls) -> Optional[Callable]:
        """
        Returns
        -------
        Optional[Callable]
            The criterion function used to train the model.
        """
        pass

    @classmethod
    @abstractmethod
    def sparsifying_trainer(cls, compress_algorithm_name: str) -> Optional[Callable[[Module, Optimizer, Callable, int], None]]:
        """
        The trainer is used in sparsifying process.

        Parameters
        ----------
        compress_algorithm_name: str
            The name of pruner and quantizer, i.e. 'level', 'l1', 'qat'.

        Returns
        -------
        Optional[Callable[[Module, Optimizer, Callable, int], None]]
            Used to train model in compress stage, include `model, optimizer, criterion, current_epoch` as function arguments.
        """
        pass

    @classmethod
    @abstractmethod
    def post_compress_finetuning_trainer(cls, compress_algorithm_name: str) -> Optional[Callable[[Module, Optimizer, Callable, int], None]]:
        """
        The trainer is used in post-compress finetuning process.

        Parameters
        ----------
        compress_algorithm_name: str
            The name of pruner and quantizer, i.e. 'level', 'l1', 'qat'.

        Returns
        -------
        Optional[Callable[[Module, Optimizer, Callable, int], None]]
            Used to train model in finetune stage, include `model, optimizer, criterion, current_epoch` as function arguments.
        """
        pass

    @classmethod
    @abstractmethod
    def post_compress_finetuning_epochs(cls, compress_algorithm_name: str) -> int:
        """
        The epochs in post-compress finetuning process.

        Parameters
        ----------
        compress_algorithm_name: str
            The name of pruner and quantizer, i.e. 'level', 'l1', 'qat'.

        Returns
        -------
        int
            The finetuning epoch number.
        """
        pass
