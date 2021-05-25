# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch.nn as nn
from nni.nas.pytorch.mutables import Mutable, MutableScope, InputChoice
from nni.nas.pytorch.utils import StructuredMutableTreeNode

logger = logging.getLogger(__name__)


class BaseMutator(nn.Module):
    """
    A mutator is responsible for mutating a graph by obtaining the search space from the network and implementing
    callbacks that are called in ``forward`` in mutables.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to apply mutator on.
    """

    def __init__(self, model):
        super().__init__()
        self.__dict__["model"] = model
        self._structured_mutables = self._parse_search_space(self.model)

    def _parse_search_space(self, module, root=None, prefix="", memo=None, nested_detection=None):
        if memo is None:
            memo = set()
        if root is None:
            root = StructuredMutableTreeNode(None)
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
                if isinstance(module, InputChoice):
                    for k in module.choose_from:
                        if k != InputChoice.NO_KEY and k not in [m.key for m in memo if isinstance(m, Mutable)]:
                            raise RuntimeError("'{}' required by '{}' not found in keys that appeared before, and is not NO_KEY."
                                               .format(k, module.key))
            for name, submodule in module._modules.items():
                if submodule is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                self._parse_search_space(submodule, root, submodule_prefix, memo=memo,
                                         nested_detection=nested_detection)
        return root

    @property
    def mutables(self):
        """
        A generator of all modules inheriting :class:`~nni.nas.pytorch.mutables.Mutable`.
        Modules are yielded in the order that they are defined in ``__init__``.
        For mutables with their keys appearing multiple times, only the first one will appear.
        """
        return self._structured_mutables

    @property
    def undedup_mutables(self):
        return self._structured_mutables.traverse(deduplicate=False)

    def forward(self, *inputs):
        """
        Warnings
        --------
        Don't call forward of a mutator.
        """
        raise RuntimeError("Forward is undefined for mutators.")

    def __setattr__(self, name, value):
        if name == "model":
            raise AttributeError("Attribute `model` can be set at most once, and you shouldn't use `self.model = model` to "
                                 "include you network, as it will include all parameters in model into the mutator.")
        return super().__setattr__(name, value)

    def enter_mutable_scope(self, mutable_scope):
        """
        Callback when forward of a MutableScope is entered.

        Parameters
        ----------
        mutable_scope : MutableScope
            The mutable scope that is entered.
        """
        pass

    def exit_mutable_scope(self, mutable_scope):
        """
        Callback when forward of a MutableScope is exited.

        Parameters
        ----------
        mutable_scope : MutableScope
            The mutable scope that is exited.
        """
        pass

    def on_forward_layer_choice(self, mutable, *args, **kwargs):
        """
        Callbacks of forward in LayerChoice.

        Parameters
        ----------
        mutable : nni.nas.pytorch.mutables.LayerChoice
            Module whose forward is called.
        args : list of torch.Tensor
            The arguments of its forward function.
        kwargs : dict
            The keyword arguments of its forward function.

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            Output tensor and mask.
        """
        raise NotImplementedError

    def on_forward_input_choice(self, mutable, tensor_list):
        """
        Callbacks of forward in InputChoice.

        Parameters
        ----------
        mutable : nni.nas.pytorch.mutables.InputChoice
            Mutable that is called.
        tensor_list : list of torch.Tensor
            The arguments mutable is called with.

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            Output tensor and mask.
        """
        raise NotImplementedError

    def export(self):
        """
        Export the data of all decisions. This should output the decisions of all the mutables, so that the whole
        network can be fully determined with these decisions for further training from scratch.

        Returns
        -------
        dict
            Mappings from mutable keys to decisions.
        """
        raise NotImplementedError
