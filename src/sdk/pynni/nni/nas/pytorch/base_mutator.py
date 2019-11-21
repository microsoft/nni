import logging

from nni.nas.pytorch.mutables import Mutable, MutableScope
from nni.nas.pytorch.utils import StructuredMutableTreeNode

logger = logging.getLogger(__name__)


class BaseMutator:
    """
    A mutator is responsible for mutating a graph by obtaining the search space from the network and implementing
    callbacks that are called in ``forward`` in Mutables.
    """

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
                self._parse_search_space(submodule, root, submodule_prefix, memo=memo,
                                         nested_detection=nested_detection)

    @property
    def mutables(self):
        return self._structured_mutables

    def enter_mutable_scope(self, mutable_scope):
        """
        Callback when forward of a MutableScope is entered.

        Parameters
        ----------
        mutable_scope: MutableScope

        Returns
        -------
        None
        """
        pass

    def exit_mutable_scope(self, mutable_scope):
        """
        Callback when forward of a MutableScope is exited.

        Parameters
        ----------
        mutable_scope: MutableScope

        Returns
        -------
        None
        """
        pass

    def get_decision(self, mutable):
        """
        Retrieve the decision of a mutable. Type of decisions might be expecting different types of mutables.
        For example, decision for a layer choice or an input choice is a mask. A customized mutable might require a
        different type of decision, or never ask one.

        Parameters
        ----------
        mutable: Mutable

        Returns
        -------
        any
        """
        raise NotImplementedError

    def on_forward_layer_choice(self, mutable, *inputs):
        """
        Callbacks of forward in LayerChoice.

        Parameters
        ----------
        mutable: LayerChoice
        inputs: list of torch.Tensor

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            output tensor and mask
        """
        raise NotImplementedError

    def on_forward_input_choice(self, mutable, tensor_list):
        """
        Callbacks of forward in InputChoice.

        Parameters
        ----------
        mutable: InputChoice
        tensor_list: list of torch.Tensor

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            output tensor and mask
        """
        raise NotImplementedError

    def export(self):
        """
        Export the data of all decisions. This should output the decisions of all the mutables, so that the whole
        network can be fully determined with these decisions for further training from scratch.

        Returns
        -------
        dict
        """
        raise NotImplementedError
