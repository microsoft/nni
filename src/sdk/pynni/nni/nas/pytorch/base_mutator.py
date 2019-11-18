import logging

import torch.nn as nn

from nni.nas.pytorch.mutables import Mutable

logger = logging.getLogger(__name__)


class BaseMutator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__["model"] = model
        self.before_parse_search_space()
        self._parse_search_space()
        self.after_parse_search_space()

    def before_parse_search_space(self):
        pass

    def after_parse_search_space(self):
        pass

    def _parse_search_space(self):
        for name, mutable, _ in self.named_mutables(distinct=False):
            mutable.name = name
            mutable.set_mutator(self)

    def named_mutables(self, root=None, distinct=True):
        if root is None:
            root = self.model
        # if distinct is true, the method will filter out those with duplicated keys
        key2module = dict()
        for name, module in root.named_modules():
            if isinstance(module, Mutable):
                module_distinct = False
                if module.key in key2module:
                    assert key2module[module.key].similar(module), \
                        "Mutable \"{}\" that share the same key must be similar to each other".format(module.key)
                else:
                    module_distinct = True
                    key2module[module.key] = module
                if distinct:
                    if module_distinct:
                        yield name, module
                else:
                    yield name, module, module_distinct

    def __setattr__(self, key, value):
        if key in ["model", "net", "network"]:
            logger.warning("Think twice if you are including the network into mutator.")
        return super().__setattr__(key, value)

    def forward(self, *inputs):
        raise NotImplementedError("Mutator is not forward-able")

    def enter_mutable_scope(self, mutable_scope):
        pass

    def exit_mutable_scope(self, mutable_scope):
        pass

    def on_forward_layer_choice(self, mutable, *inputs):
        raise NotImplementedError

    def on_forward_input_choice(self, mutable, tensor_list, tags):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
