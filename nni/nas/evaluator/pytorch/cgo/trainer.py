# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytorch_lightning as pl
from pytorch_lightning.strategies import SingleDeviceStrategy


class BypassStrategy(SingleDeviceStrategy):
    strategy_name = "single_device"

    def model_to_device(self) -> None:
        pass


class Trainer(pl.Trainer):
    """
    Trainer for cross-graph optimization.

    Parameters
    ----------
    use_cgo : bool
        Whether cross-graph optimization (CGO) is used.
        If it is True, CGO will manage device placement.
        Any device placement from pytorch lightning will be bypassed.
        default: False
    trainer_kwargs : dict
        Optional keyword arguments passed to trainer. See
        `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for details.
    """

    def __init__(self, use_cgo=False, **trainer_kwargs):
        if use_cgo:
            if "accelerator" in trainer_kwargs:
                raise ValueError("accelerator should not be set when cross-graph optimization is enabled.")

            if 'strategy' in trainer_kwargs:
                raise ValueError("cgo.trainer does not support specifying strategy")
            trainer_kwargs['strategy'] = BypassStrategy()

        super().__init__(**trainer_kwargs)
