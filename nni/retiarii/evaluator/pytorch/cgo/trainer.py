

import pytorch_lightning as pl
from ....serializer import serialize_cls
from .accelerator import BypassAccelerator


@serialize_cls
class Trainer(pl.Trainer):
    def __init__(self, use_cgo=False, **trainer_kwargs):
        if use_cgo:
            if "accelerator" in trainer_kwargs:
                raise ValueError("accelerator should not be set when cross-graph optimization is enabled.")
            trainer_kwargs['accelerator'] = BypassAccelerator(device='cpu')

        super().__init__(**trainer_kwargs)
