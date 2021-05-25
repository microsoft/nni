# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Optional, Callable

from torch.nn import Module
from torch.optim import Optimizer


class AbstractExecutionEngine(ABC):
    @classmethod
    @abstractmethod
    def trial_execute_compress(cls):
        """
        Execute the compressing trial
        """
        pass


class AbstractAutoCompressModule(ABC):
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
            Model to be compress
        """
        pass

    @classmethod
    @abstractmethod
    def evaluator(cls) -> Callable[[Module], float]:
        """
        Returns
        -------
        function
            A function used to evaluate the compressed model, return a scalar
        """
        pass

    @classmethod
    @abstractmethod
    def optimizer(cls) -> Optional[Optimizer]:
        """
        Returns
        -------
        torch.optim.Optimizer
            Optimizer used to train the model in compressing process.
        """
        pass

    @classmethod
    @abstractmethod
    def criterion(cls) -> Optional[Callable]:
        """
        Returns
        -------
        Optional[Callable]
            The criterion used to train the model in compressing process.
        """
        pass

    @classmethod
    @abstractmethod
    def sparsifying_trainer(cls, compressor_type: str, algorithm_name: str) -> Optional[Callable[[Module, Optimizer], None]]:
        """
        The trainer is used in sparsifying process.

        Parameters
        ----------
        compressor_type: str
            Support 'pruner' and 'quantizer'
        algorithm_name: str
            The name of pruner and quantizer, i.e. 'level', 'l1', 'qat'

        Returns
        -------
        function
            Used to train model in compress stage
        """
        pass

    @classmethod
    @abstractmethod
    def post_compress_finetuning_trainer(cls, compressor_type: str, algorithm_name: str) -> Optional[Callable[[Module, Optimizer], None]]:
        """
        The trainer is used in post-compress finetuning process.

        Parameters
        ----------
        compressor_type: str
            Support 'pruner' and 'quantizer'
        algorithm_name: str
            The name of pruner and quantizer, i.e. 'level', 'l1', 'qat'

        Returns
        -------
        function
            Used to train model in finetune stage
        """
        pass
