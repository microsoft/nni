# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging

from nni.nas.pytorch.callbacks import LRSchedulerCallback
from nni.algorithms.nas.pytorch.darts import DartsTrainer
from nni.nas.pytorch.trainer import BaseTrainer, TorchTensorEncoder

from .mutator import PdartsMutator

logger = logging.getLogger(__name__)


class PdartsTrainer(BaseTrainer):
    """
    This trainer implements the PDARTS algorithm.
    PDARTS bases on DARTS algorithm, and provides a network growth approach to find deeper and better network.
    This class relies on pdarts_num_layers and pdarts_num_to_drop parameters to control how network grows.
    pdarts_num_layers means how many layers more than first epoch.
    pdarts_num_to_drop means how many candidate operations should be dropped in each epoch.
        So that the grew network can in similar size.
    """

    def __init__(self, model_creator, init_layers, metrics,
                 num_epochs, dataset_train, dataset_valid,
                 pdarts_num_layers=[0, 6, 12], pdarts_num_to_drop=[3, 2, 1],
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None, callbacks=None, unrolled=False):
        super(PdartsTrainer, self).__init__()
        self.model_creator = model_creator
        self.init_layers = init_layers
        self.pdarts_num_layers = pdarts_num_layers
        self.pdarts_num_to_drop = pdarts_num_to_drop
        self.pdarts_epoch = len(pdarts_num_to_drop)
        self.darts_parameters = {
            "metrics": metrics,
            "num_epochs": num_epochs,
            "dataset_train": dataset_train,
            "dataset_valid": dataset_valid,
            "batch_size": batch_size,
            "workers": workers,
            "device": device,
            "log_frequency": log_frequency,
            "unrolled": unrolled
        }
        self.callbacks = callbacks if callbacks is not None else []

    def train(self):

        switches = None
        for epoch in range(self.pdarts_epoch):

            layers = self.init_layers+self.pdarts_num_layers[epoch]
            model, criterion, optim, lr_scheduler = self.model_creator(layers)
            self.mutator = PdartsMutator(model, epoch, self.pdarts_num_to_drop, switches)

            for callback in self.callbacks:
                callback.build(model, self.mutator, self)
                callback.on_epoch_begin(epoch)

            darts_callbacks = []
            if lr_scheduler is not None:
                darts_callbacks.append(LRSchedulerCallback(lr_scheduler))

            self.trainer = DartsTrainer(model, mutator=self.mutator, loss=criterion, optimizer=optim,
                                        callbacks=darts_callbacks, **self.darts_parameters)
            logger.info("start pdarts training epoch %s...", epoch)

            self.trainer.train()

            switches = self.mutator.drop_paths()

            for callback in self.callbacks:
                callback.on_epoch_end(epoch)

    def validate(self):
        self.trainer.validate()

    def export(self, file):
        mutator_export = self.mutator.export()
        with open(file, "w") as f:
            json.dump(mutator_export, f, indent=2, sort_keys=True, cls=TorchTensorEncoder)

    def checkpoint(self):
        raise NotImplementedError("Not implemented yet")
