import json
import logging
import os

import torch

_logger = logging.getLogger(__name__)


class Callback:

    def __init__(self):
        self.model = None
        self.mutator = None
        self.trainer = None

    def build(self, model, mutator, trainer):
        self.model = model
        self.mutator = mutator
        self.trainer = trainer

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_batch_begin(self, epoch):
        pass

    def on_batch_end(self, epoch):
        pass


class LearningRateScheduler(Callback):
    def __init__(self, scheduler, mode="epoch"):
        super().__init__()
        assert mode == "epoch"
        self.scheduler = scheduler
        self.mode = mode

    def on_epoch_end(self, epoch):
        self.scheduler.step()


class ArchitectureCheckpoint(Callback):
    class TorchTensorEncoder(json.JSONEncoder):
        def default(self, o):  # pylint: disable=method-hidden
            if isinstance(o, torch.Tensor):
                olist = o.tolist()
                if "bool" not in o.type().lower() and all(map(lambda d: d == 0 or d == 1, olist)):
                    _logger.warning("Every element in %s is either 0 or 1. "
                                    "You might consider convert it into bool.", olist)
                return olist
            return super().default(o)

    def __init__(self, checkpoint_dir, every="epoch"):
        super().__init__()
        assert every == "epoch"
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _export_to_file(self, file):
        mutator_export = self.mutator.export()
        with open(file, "w") as f:
            json.dump(mutator_export, f, indent=2, sort_keys=True, cls=self.TorchTensorEncoder)

    def on_epoch_end(self, epoch):
        self._export_to_file(os.path.join(self.checkpoint_dir, "epoch_{}.json".format(epoch)))
