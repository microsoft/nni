# import logging

from nni.nas.pytorch.darts import CnnCell, DartsTrainer
from nni.nas.pytorch.modules import RankedModule
from nni.nas.pytorch.mutables import LayerChoice
from nni.nas.pytorch.trainer import Trainer
from nni.nas.utils import AverageMeterGroup

from .mutator import PdartsMutator

# _logger = logging.getLogger()
# _logger.setLevel(logging.INFO)


class PdartsTrainer(Trainer):

    def __init__(self, model_creator, metrics,
                 num_epochs, dataset_train, dataset_valid,
                 layers=5, n_nodes=4,
                 pdarts_num_layers=[0, 6, 12],
                 pdarts_num_to_drop=[3, 2, 2], mutator=None,
                 batch_size=64, workers=4, device=None, log_frequency=None):
        self.model_creator = model_creator
        self.layers = layers
        self.n_nodes = n_nodes
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
        n_nodes = self.n_nodes
        switches = None
        for epoch in range(self.pdarts_epoch):

            layers = self.layers+self.pdarts_num_layers[epoch]
            model, loss, model_optim, lr_scheduler = self.model_creator(
                layers, n_nodes)
            mutator = PdartsMutator(
                model, epoch, self.pdarts_num_to_drop, switches)

            trainer = DartsTrainer(model, loss=loss, model_optim=model_optim,
                                   lr_scheduler=lr_scheduler, mutator=mutator,
                                   **self.darts_parameters)
            print("start training")

            with mutator.forward_pass():
                trainer.train()

            # for key in mutator.choices:
            #     item = mutator.choices[key]
            #     print("key: %s, %s" % (key, item.cpu().data))

            # with open('log/parameters_%d.txt' % epoch, "w") as f:
            #     f.write(str(model.parameters))

            switches = mutator.drop_paths()

    def export(self):
        pass
