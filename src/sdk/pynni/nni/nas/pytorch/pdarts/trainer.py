# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint, LearningRateScheduler)
from nni.nas.pytorch.darts import DartsTrainer
from nni.nas.pytorch.trainer import BaseTrainer

from .mutator import PdartsMutator


class PdartsTrainer(BaseTrainer):

    def __init__(self, model_creator, layers, metrics,
                 num_epochs, dataset_train, dataset_valid,
                 pdarts_num_layers=[0, 6, 12], pdarts_num_to_drop=[3, 2, 2],
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None):
        super(PdartsTrainer, self).__init__()
        self.model_creator = model_creator
        self.layers = layers
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
            "log_frequency": log_frequency
        }

    def train(self):
        layers = self.layers
        switches = None
        for epoch in range(self.pdarts_epoch):

            layers = self.layers+self.pdarts_num_layers[epoch]
            model, criterion, optim, lr_scheduler = self.model_creator(layers)

            self.mutator = PdartsMutator(model, epoch, self.pdarts_num_to_drop, switches)

            self.trainer = DartsTrainer(model, mutator=self.mutator, loss=criterion, optimizer=optim, callbacks=[LearningRateScheduler(
                lr_scheduler), ArchitectureCheckpoint("./checkpoints")], **self.darts_parameters)
            print("start pdarts training %s..." % epoch)

            self.trainer.train()

            switches = self.mutator.drop_paths()

    def validate(self):
        self.model.validate()

    def export(self):
        self.mutator.export()

    def checkpoint(self):
        raise NotImplementedError("Not implemented yet")
