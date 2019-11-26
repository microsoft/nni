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
        pdarts_epoch_index = 1
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

                switches = self.switches.get(mutable.key, [True for j in range(mutable.length)])

                for index in range(len(switches)-1, -1, -1):
                    if switches[index] == False:
                        del(mutable.choices[index])
                        mutable.length -= 1
                logger.info("1. choices key %s %s", mutable.key, self.choices[mutable.key])
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length + 1))
                self.switches[mutable.key] = switches
                logger.info("2. choices %s", self.choices[mutable.key])

        for module in self.model.modules():
            if isinstance(module, LayerChoice):
                switches = self.switches.get(module.key)
                choices = self.choices[module.key]
                if len(module.choices) > len(choices):
                    logger.info("1. module choices count key %s %s", module.key, module.choices)
                    for index in range(len(switches)-1, -1, -1):
                        if switches[index] == False:
                            del(module.choices[index])
                            module.length -= 1
                    logger.info("2. module choices %s", module.choices)

    def drop_paths(self):
        for key in self.switches:
            prob = F.softmax(self.choices[key], dim=-1).data.cpu().numpy()

            switches = self.switches[key]
            idxs = []
            for j in range(len(switches)):
                if switches[j]:
                    idxs.append(j)
            if self.pdarts_epoch_index == len(self.pdarts_num_to_drop) - 1:
                # for the last stage, drop all Zero operations
                drop = self.get_min_k_no_zero(prob, idxs, self.pdarts_num_to_drop[self.pdarts_epoch_index])
            else:
                drop = self.get_min_k(prob, self.pdarts_num_to_drop[self.pdarts_epoch_index])

            for idx in drop:
                switches[idxs[idx]] = False
        return self.switches

    def get_min_k(self, input_in, k):
        index = []
        for _ in range(k):
            idx = np.argmin(input)
            index.append(idx)

        return index

    def get_min_k_no_zero(self, w_in, idxs, k):
        w = copy.deepcopy(w_in)
        index = []
        if 0 in idxs:
            zf = True
        else:
            zf = False
        if zf:
            w = w[1:]
            index.append(0)
            k = k - 1
        for _ in range(k):
            idx = np.argmin(w)
            w[idx] = 1
            if zf:
                idx = idx + 1
            index.append(idx)
        return index
