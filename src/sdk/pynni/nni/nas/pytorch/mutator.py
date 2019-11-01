import logging

from torch import nn as nn

from nni.nas.pytorch.mutables import PyTorchMutable
from nni.nas.utils import to_snake_case

logger = logging.getLogger(__name__)


class PyTorchMutator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.before_build(model)
        self.parse_search_space(model)
        self.after_build(model)

    def before_build(self, model):
        pass

    def after_build(self, model):
        pass

    def named_mutables(self, model):
        # if distinct is true, the method will filter out those with duplicated keys
        key2module = dict()
        for name, module in model.named_modules():
            if isinstance(module, PyTorchMutable):
                distinct = False
                if module.key in key2module:
                    assert key2module[module.key].similar(module), "Mutable that share the same key must be similar " \
                                                                    "to each other"
                else:
                    distinct = True
                    key2module[module.key] = module
                yield name, module, distinct

    def __setattr__(self, key, value):
        if key in ["model", "net", "network"]:
            logger.warning("Think twice if you are including the network into mutator.")
        return super().__setattr__(key, value)

    def parse_search_space(self, model):
        for name, mutable, distinct in self.named_mutables(model):
            mutable.name = name
            mutable.set_mutator(self)
            if not distinct:
                continue
            init_method_name = "on_init_{}".format(to_snake_case(mutable.__class__.__name__))
            if hasattr(self, init_method_name) and callable(getattr(self, init_method_name)):
                getattr(self, init_method_name)(mutable)
            else:
                # fallback to general init
                self.on_init_general(mutable)

    def on_init_general(self, mutable):
        pass

    def on_forward_general(self, mutable, *inputs):
        raise NotImplementedError("Forward has to be implemented")

    def on_forward(self, mutable, *inputs):
        """Callback on forwarding a mutable"""
        forward_method_name = "on_forward_{}".format(to_snake_case(mutable.__class__.__name__))
        if hasattr(self, forward_method_name) and callable(getattr(self, forward_method_name)):
            return getattr(self, forward_method_name)(mutable, *inputs)
        else:
            # fallback to general forward
            return self.on_forward_general(mutable, *inputs)

    def forward(self, *inputs):
        raise NotImplementedError("Mutator is not forward-able")
