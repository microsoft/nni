import logging

from nni.nas.utils import AverageMeterGroup
from nni.nas.pytorch.darts import DartsTrainer
from nni.nas.pytorch import LayerChoice

from .mutator import PdartsMutator

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

class LayerChoiceSwitches():
    def __init__(self):
        self.name = ""
        self.isReduce = False
        self.switches = []

    def __repr__(self):
        return "name: {}, isReduce: {}, switches: {}\n".format(self.name, self.isReduce, self.switches)

class PdartsTrainer(DartsTrainer):

    def __init__(self, model, loss, metrics,
                 model_optim, lr_scheduler, num_epochs, dataset_train, dataset_valid,
                 pdarts_epoch=3, pdarts_num_to_drop=[3, 2, 2],
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None):
        self.mutator = PdartsMutator(model)
        super(PdartsTrainer, self).__init__(model, loss, metrics, model_optim, lr_scheduler, num_epochs, dataset_train,
                                            dataset_valid, mutator=None, batch_size=64, workers=4, device=None, log_frequency=None)
        self.pdarts_epoch = pdarts_epoch
        self.pdarts_num_to_drop = pdarts_num_to_drop

        switches = []
        count = 0
        for key, value in model.named_modules():

            if type(value) == LayerChoice:
                count += 1
                layerSwitches = LayerChoiceSwitches()
                value.name = "named_layer_choice_%d" % count
                layerSwitches.name = value.name
                layerSwitches.switches = [True for j in range(len(value.choices))]
                switches.append(layerSwitches)
        self.switches = switches
        print(self.switches)
        if pdarts_epoch != len(pdarts_num_to_drop):
            raise Exception("pdarts_num_to_drop '%s' must be array, and it's length(%d) should be the same as pdarts_epoch(%d).",
                            pdarts_num_to_drop, len(pdarts_num_to_drop), pdarts_epoch)

    def train(self):
        for epoch in range(self.pdarts_epoch):
            print("start training")
            super().train()

            for key, value in self.model.named_modules():
                if type(value) == LayerChoice:
                    print("key: %s, name %s" % (key, value.name))

            with open('log/parameters_%d.txt' % epoch, "w") as f:
                f.write(str(self.model.parameters))
