import torch
from torch import nn as nn
from torch.nn import functional as F

from nni.nas.pytorch.mutables import LayerChoice
from nni.nas.pytorch.mutator import Mutator
from .scope import DartsNode


class DartsMutator(Mutator):

    def after_parse_search_space(self):
        self.choices = nn.ParameterDict()
        for _, mutable in self.named_mutables():
            if isinstance(mutable, LayerChoice):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(len(mutable) + 1))

    def on_calc_layer_choice_mask(self, mutable: LayerChoice):
        return F.softmax(self.choices[mutable.key], dim=-1)[:-1]

    def export(self):
        result = super().export()
        for _, darts_node in self.named_mutables():
            if isinstance(darts_node, DartsNode):
                keys, edges_max = [], []  # key of all the layer choices in current node, and their best edge weight
                for _, choice in self.named_mutables(darts_node):
                    if isinstance(choice, LayerChoice):
                        keys.append(choice.key)
                        max_val, index = torch.max(result[choice.key], 0)
                        edges_max.append(max_val)
                        result[choice.key] = F.one_hot(index, num_classes=len(result[choice.key])).view(-1).bool()
                _, topk_edge_indices = torch.topk(torch.tensor(edges_max).view(-1), darts_node.limitation)  # pylint: disable=not-callable
                for i, key in enumerate(keys):
                    if i not in topk_edge_indices:
                        result[key] = torch.zeros_like(result[key])
        return result
