import abc
from typing import Any


class BaseTrainer(abc.ABC):
    """
    In this version, we plan to write our own trainers instead of using PyTorch-lightning, to
    ease the burden to integrate our optmization with PyTorch-lightning, a large part of which is
    opaque to us.

    We will try to align with PyTorch-lightning name conversions so that we can easily migrate to
    PyTorch-lightning in the future.

    Currently, our trainer = LightningModule + LightningTrainer. We might want to separate these two things
    in future.

    Trainer has a ``fit`` function with no return value. Intermediate results and final results should be
    directly sent via ``nni.report_intermediate_result()`` and ``nni.report_final_result()`` functions.
    """

    @abc.abstractmethod
    def fit(self) -> None:
        pass


class BaseOneShotTrainer(BaseTrainer):
    """
    Build many (possibly all) architectures into a full graph, search (with train) and export the best.

    It has an extra ``export`` function that exports an object representing the final searched architecture.
    """

    @abc.abstractmethod
    def export(self) -> Any:
        pass
