from nni.nas.pytorch.darts import DartsTrainer
from nni.nas.pytorch.trainer import Trainer

from .mutator import PdartsMutator


class PdartsTrainer(Trainer):

    def __init__(self, model_creator, metrics, num_epochs, dataset_train, dataset_valid,
                 layers=5, n_nodes=4, pdarts_num_layers=[0, 6, 12], pdarts_num_to_drop=[3, 2, 2],
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None):
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
            model, loss, model_optim, _ = self.model_creator(
                layers, n_nodes)
            mutator = PdartsMutator(model, epoch, self.pdarts_num_to_drop, switches)  # pylint: disable=too-many-function-args

            self.trainer = DartsTrainer(model, loss=loss, optimizer=model_optim,
                                        mutator=mutator, **self.darts_parameters)
            print("start pdrats training %s..." % epoch)

            self.trainer.train()

            # with open('log/parameters_%d.txt' % epoch, "w") as f:
            #     f.write(str(model.parameters))

            switches = mutator.drop_paths()

    def export(self):
        if (self.trainer is not None) and hasattr(self.trainer, "export"):
            self.trainer.export()
