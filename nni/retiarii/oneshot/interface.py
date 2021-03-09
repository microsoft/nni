# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Any


class BaseOneShotTrainer(abc.ABC):
    """
    Build many (possibly all) architectures into a full graph, search (with train) and export the best.

    One-shot trainer has a ``fit`` function with no return value. Trainers should fit and search for the best architecture.
    Currently, all the inputs of trainer needs to be manually set before fit (including the search space, data loader
    to use training epochs, and etc.).

    It has an extra ``export`` function that exports an object representing the final searched architecture.
    """

    @abc.abstractmethod
    def fit(self) -> None:
        pass

    @abc.abstractmethod
    def export(self) -> Any:
        pass
