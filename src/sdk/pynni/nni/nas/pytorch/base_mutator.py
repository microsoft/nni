import logging

from nni.nas.pytorch.mutables import Mutable, MutableScope
from nni.nas.pytorch.utils import StructuredMutableTreeNode

logger = logging.getLogger(__name__)


class BaseMutator:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._structured_mutables = StructuredMutableTreeNode(None)
        self._parse_search_space(self.model, self._structured_mutables)

    def _parse_search_space(self, module, root, prefix="", memo=None, nested_detection=None):
        if memo is None:
            memo = set()
        if module not in memo:
            memo.add(module)
            if isinstance(module, Mutable):
                if nested_detection is not None:
                    raise RuntimeError("Cannot have nested search space. Error at {} in {}"
                                       .format(module, nested_detection))
                module.name = prefix
                module.set_mutator(self)
                root = root.add_child(module)
                if not isinstance(module, MutableScope):
                    nested_detection = module
            for name, submodule in module._modules.items():
                if submodule is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                self._parse_search_space(submodule, root, submodule_prefix, memo=memo, nested_detection=nested_detection)

    def enter_mutable_scope(self, mutable_scope):
        pass

    def exit_mutable_scope(self, mutable_scope):
        pass

    def get_decision(self, mutable):
        raise NotImplementedError

    def on_forward_layer_choice(self, mutable, *inputs):
        raise NotImplementedError

    def on_forward_input_choice(self, mutable, tensor_list):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
