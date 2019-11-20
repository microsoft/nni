import copy

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from nni.nas.pytorch.darts import DartsMutator
from nni.nas.pytorch.mutables import LayerChoice


class PdartsMutator(DartsMutator):

    def __init__(self, model, pdarts_epoch_index, pdarts_num_to_drop, switches=None):
        self.pdarts_epoch_index = pdarts_epoch_index
        self.pdarts_num_to_drop = pdarts_num_to_drop
        if switches is None:
            self.switches = {}
        else:
            self.switches = switches

        super(PdartsMutator, self).__init__(model)

    def after_parse_search_space(self):
        self.choices = nn.ParameterDict()

        for _, mutable in self.named_mutables():
            if isinstance(mutable, LayerChoice):

                switches = self.switches.get(
                    mutable.key, [True for j in range(mutable.length)])

                for index in range(len(switches)-1, -1, -1):
                    if switches[index] == False:
                        del(mutable.choices[index])
                        mutable.length -= 1

                self.switches[mutable.key] = switches
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(len(mutable) + 1))

    def on_calc_layer_choice_mask(self, mutable: LayerChoice):
        return F.softmax(self.choices[mutable.key], dim=-1)

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
