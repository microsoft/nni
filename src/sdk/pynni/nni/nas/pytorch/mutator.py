import logging
from collections import OrderedDict

from torch import nn as nn

from nni.nas.pytorch.mutables import Mutable
from nni.nas.utils import to_snake_case

logger = logging.getLogger(__name__)


class Mutator(nn.Module):
    def __init__(self, model):
        super().__init__()

        self._on_init_hooks = {}
        self._on_forward_hooks = {}

        self.before_build(model)
        self.parse_search_space(model)
        self.after_build(model)

    def before_build(self, model):
        pass

    def after_build(self, model):
        pass

    def register_on_init_hook(self, mutable_type, hook):
        hooks = self._on_init_hooks.get(mutable_type, [])
        hooks.append(hook)
        self._on_init_hooks[mutable_type] = hooks

    def register_on_forward_hook(self, mutable_type, hook):
        hooks = self._on_forward_hooks.get(mutable_type, [])
        hooks.append(hook)
        self._on_forward_hooks[mutable_type] = hooks

    def named_mutables(self, model):
        # if distinct is true, the method will filter out those with duplicated keys
        key2module = dict()
        for name, module in model.named_modules():
            if isinstance(module, Mutable):
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
            logger.warning(
                "Think twice if you are including the network into mutator.")
        return super().__setattr__(key, value)

    def parse_search_space(self, model):
        for name, mutable, distinct in self.named_mutables(model):
            mutable.name = name
            mutable.set_mutator(self)
            if not distinct:
                continue

            hooks = self._on_init_hooks.get(type(mutable))
            if hooks is not None:
                for hook in hooks:
                    hook(mutable)

    def on_forward_general(self, mutable, *inputs):
        pass

    def on_forward(self, mutable, *inputs):
        """Callback on forwarding a mutable"""

        hooks = self._on_forward_hooks.get(type(mutable))
        if hooks is not None:
            for hook in hooks:
                hook(mutable, *inputs)
        else:
            # fallback to general forward
            return self.on_forward_general(mutable, *inputs)

    def forward(self, *inputs):
        raise NotImplementedError("Mutator is not forward-able")
