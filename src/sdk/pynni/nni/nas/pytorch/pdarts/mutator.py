# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nni.nas.pytorch.darts import DartsMutator
from nni.nas.pytorch.mutables import LayerChoice


class PdartsMutator(DartsMutator):

    def __init__(self, model, pdarts_epoch_index, pdarts_num_to_drop, switches={}):
        self.pdarts_epoch_index = pdarts_epoch_index
        self.pdarts_num_to_drop = pdarts_num_to_drop
        if switches is None:
            self.switches = {}
        else:
            self.switches = switches

        super(PdartsMutator, self).__init__(model)

        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):

                switches = self.switches.get(mutable.key, [True for j in range(mutable.length+1)])
                choices = self.choices[mutable.key]

                for index in range(len(switches)-2, -1, -1):
                    if switches[index] == False:
                        del(mutable.choices[index])
                        mutable.length -= 1
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length + 1))
                self.switches[mutable.key] = switches

        # update LayerChoice instances
        for module in self.model.modules():
            if isinstance(module, LayerChoice):
                switches = self.switches.get(module.key)
                choices = self.choices[module.key]
                if len(module.choices) > len(choices):
                    for index in range(len(switches)-2, -1, -1):
                        if switches[index] == False:
                            del(module.choices[index])
                            module.length -= 1

    def sample_final(self):
        results = super().sample_final()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                trained_result = results[mutable.key]
                trained_index = 0
                switches = self.switches[mutable.key]
                result = torch.Tensor(switches[:-1]).bool()
                for index in range(len(result)):
                    if result[index]:
                        result[index] = trained_result[trained_index]
                        trained_index += 1
                results[mutable.key] = result
        return results

    def drop_paths(self):
        all_switches = copy.deepcopy(self.switches)
        for key in all_switches:
            switches = all_switches[key]
            idxs = []
            for j in range(len(switches)-1):
                if switches[j]:
                    idxs.append(j)
            sorted_weights = self.choices[key].data.cpu().numpy()[:-1]
            drop = np.argsort(sorted_weights)[:self.pdarts_num_to_drop[self.pdarts_epoch_index]]
            for idx in drop:
                switches[idxs[idx]] = False
        return all_switches
