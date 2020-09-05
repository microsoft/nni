# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tensorflow.keras import Model

from .mutables import Mutable, MutableScope, InputChoice
from .utils import StructuredMutableTreeNode


class BaseMutator(Model):
    def __init__(self, model):
        super().__init__()
        self.__dict__['model'] = model
        self._structured_mutables = self._parse_search_space(self.model)

    def _parse_search_space(self, module, root=None, prefix='', memo=None, nested_detection=None):
        if memo is None:
            memo = set()
        if root is None:
            root = StructuredMutableTreeNode(None)
        if module not in memo:
            memo.add(module)
            if isinstance(module, Mutable):
                if nested_detection is not None:
                    raise RuntimeError('Cannot have nested search space. Error at {} in {}'
                                       .format(module, nested_detection))
                module.name = prefix
                module.set_mutator(self)
                root = root.add_child(module)
                if not isinstance(module, MutableScope):
                    nested_detection = module
                if isinstance(module, InputChoice):
                    for k in module.choose_from:
                        if k != InputChoice.NO_KEY and k not in [m.key for m in memo if isinstance(m, Mutable)]:
                            raise RuntimeError('"{}" required by "{}" not found in keys that appeared before, and is not NO_KEY.'
                                               .format(k, module.key))
            for submodule in module.layers:
                if not isinstance(submodule, Model):
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + submodule.name
                self._parse_search_space(submodule, root, submodule_prefix, memo=memo, nested_detection=nested_detection)
        return root

    @property
    def mutables(self):
        return self._structured_mutables

    def undedup_mutables(self):
        return self._structured_mutables.traverse(deduplicate=False)

    def call(self, *inputs):
        raise RuntimeError('Call is undefined for mutators.')

    def __setattr__(self, name, value):
        if name == 'model':
            raise AttributeError("Attribute `model` can be set at most once, and you shouldn't use `self.model = model` to "
                                 "include your network, as it will include all parameters in model into the mutator.")
        return super().__setattr__(name, value)

    def enter_mutable_scope(self, mutable_scope):
        pass

    def exit_mutable_scope(self, mutable_scope):
        pass

    def on_forward_layer_choice(self, mutable, *inputs):
        raise NotImplementedError

    def on_forward_input_choice(self, mutable, tensor_list):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
