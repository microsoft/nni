# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.darts import DartsMutator
from nni.nas.pytorch.mutables import LayerChoice

logger = logging.getLogger(__name__)


class PdartsMutator(DartsMutator):

    def __init__(self, model, pdarts_epoch_index, pdarts_num_to_drop, switches={}):
        self.pdarts_epoch_index = pdarts_epoch_index
        self.pdarts_num_to_drop = pdarts_num_to_drop
        if switches is None:
            self.switches = {}
        else:
            self.switches = switches

        super(PdartsMutator, self).__init__(model)

        logger.info("inited choices %s", self.choices)
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):

                switches = self.switches.get(mutable.key, [True for j in range(mutable.length+1)])

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
                    logger.info("1. module choices count key %s %s", module.key, module.choices)
                    for index in range(len(switches)-2, -1, -1):
                        if switches[index] == False:
                            del(module.choices[index])
                            module.length -= 1

    def drop_paths(self):
        for key in self.switches:
            prob = F.softmax(self.choices[key], dim=-1).data.cpu().numpy()

            switches = self.switches[key]
            idxs = []
            for j in range(len(switches)):
                if switches[j]:
                    idxs.append(j)
            drop = self.get_min_k(prob, self.pdarts_num_to_drop[self.pdarts_epoch_index])

            for idx in drop:
                switches[idxs[idx]] = False
        return self.switches

    def get_min_k(self, input_in, k):
        input_copy = copy.deepcopy(input_in[:-1])

        index = []
        for _ in range(k):
            idx = np.argmin(input_copy)
            index.append(idx)
            input_copy[idx] = 1

        return index
