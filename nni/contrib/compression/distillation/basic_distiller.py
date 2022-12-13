import torch

from nni.compression.pytorch.utils.evaluator import Evaluator

from ..base.compressor import Distiller


class BasicDistiller(Distiller):
    def __init__(self, model: torch.nn.Module):
        config_list = [{'op_names': ['']}]
        super().__init__(model, config_list)

    def compress(self, epoch: ):
        return super().compress()
