# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy

import numpy as np
import torch
from torch import nn

from nni.algorithms.nas.pytorch.darts import DartsMutator
from nni.nas.pytorch.mutables import LayerChoice


class PdartsMutator(DartsMutator):
    """
    It works with PdartsTrainer to calculate ops weights,
    and drop weights in different PDARTS epochs.
    """

    def __init__(self, model, pdarts_epoch_index, pdarts_num_to_drop, switches={}):
        self.pdarts_epoch_index = pdarts_epoch_index
        self.pdarts_num_to_drop = pdarts_num_to_drop
        if switches is None:
            self.switches = {}
        else:
            self.switches = switches

        super(PdartsMutator, self).__init__(model)

        # this loop go through mutables with different keys,
        # it's mainly to update length of choices.
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):

                switches = self.switches.get(mutable.key, [True for j in range(len(mutable))])
                choices = self.choices[mutable.key]

                operations_count = np.sum(switches)
                # +1 and -1 are caused by zero operation in darts network
                # the zero operation is not in choices list in network, but its weight are in,
                # so it needs one more weights and switch for zero.
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(operations_count + 1))
                self.switches[mutable.key] = switches

        # update LayerChoice instances in model,
        # it's physically remove dropped choices operations.
        for module in self.model.modules():
            if isinstance(module, LayerChoice):
                switches = self.switches.get(module.key)
                choices = self.choices[module.key]
                if len(module) > len(choices):
                    # from last to first, so that it won't effect previous indexes after removed one.
                    for index in range(len(switches)-1, -1, -1):
                        if switches[index] == False:
                            del module[index]
                assert len(module) <= len(choices), "Failed to remove dropped choices."

    def export(self):
        # Cannot rely on super().export() because P-DARTS has deleted some of the choices and has misaligned length.
        results = super().sample_final()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                # As some operations are dropped physically,
                # so it needs to fill back false to track dropped operations.
                trained_result = results[mutable.key]
                trained_index = 0
                switches = self.switches[mutable.key]
                result = torch.Tensor(switches).bool()
                for index in range(len(result)):
                    if result[index]:
                        result[index] = trained_result[trained_index]
                        trained_index += 1
                results[mutable.key] = result
        return results

    def drop_paths(self):
        """
        This method is called when a PDARTS epoch is finished.
        It prepares switches for next epoch.
        candidate operations with False switch will be doppped in next epoch.
        """
        all_switches = copy.deepcopy(self.switches)
        for key in all_switches:
            switches = all_switches[key]
            idxs = []
            for j in range(len(switches)):
                if switches[j]:
                    idxs.append(j)
            sorted_weights = self.choices[key].data.cpu().numpy()[:-1]
            drop = np.argsort(sorted_weights)[:self.pdarts_num_to_drop[self.pdarts_epoch_index]]
            for idx in drop:
                switches[idxs[idx]] = False
        return all_switches
