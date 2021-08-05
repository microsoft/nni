# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import warnings
from collections import OrderedDict

import torch.nn as nn

from nni.nas.pytorch.utils import global_mutable_counting

logger = logging.getLogger(__name__)


class Mutable(nn.Module):
    """
    Mutable is designed to function as a normal layer, with all necessary operators' weights.
    States and weights of architectures should be included in mutator, instead of the layer itself.

    Mutable has a key, which marks the identity of the mutable. This key can be used by users to share
    decisions among different mutables. In mutator's implementation, mutators should use the key to
    distinguish different mutables. Mutables that share the same key should be "similar" to each other.

    Currently the default scope for keys is global. By default, the keys uses a global counter from 1 to
    produce unique ids.

    Parameters
    ----------
    key : str
        The key of mutable.

    Notes
    -----
    The counter is program level, but mutables are model level. In case multiple models are defined, and
    you want to have `counter` starting from 1 in the second model, it's recommended to assign keys manually
    instead of using automatic keys.
    """

    def __init__(self, key=None):
        super().__init__()
        if key is not None:
            if not isinstance(key, str):
                key = str(key)
                logger.warning("Warning: key \"%s\" is not string, converted to string.", key)
            self._key = key
        else:
            self._key = self.__class__.__name__ + str(global_mutable_counting())
        self.init_hook = self.forward_hook = None

    def __deepcopy__(self, memodict=None):
        raise NotImplementedError("Deep copy doesn't work for mutables.")

    def __call__(self, *args, **kwargs):
        self._check_built()
        return super().__call__(*args, **kwargs)

    def set_mutator(self, mutator):
        if "mutator" in self.__dict__:
            raise RuntimeError("`set_mutator` is called more than once. Did you parse the search space multiple times? "
                               "Or did you apply multiple fixed architectures?")
        self.__dict__["mutator"] = mutator

    @property
    def key(self):
        """
        Read-only property of key.
        """
        return self._key

    @property
    def name(self):
        """
        After the search space is parsed, it will be the module name of the mutable.
        """
        return self._name if hasattr(self, "_name") else self._key

    @name.setter
    def name(self, name):
        self._name = name

    def _check_built(self):
        if not hasattr(self, "mutator"):
            raise ValueError(
                "Mutator not set for {}. You might have forgotten to initialize and apply your mutator. "
                "Or did you initialize a mutable on the fly in forward pass? Move to `__init__` "
                "so that trainer can locate all your mutables. See NNI docs for more details.".format(self))


class MutableScope(Mutable):
    """
    Mutable scope marks a subgraph/submodule to help mutators make better decisions.

    If not annotated with mutable scope, search space will be flattened as a list. However, some mutators might
    need to leverage the concept of a "cell". So if a module is defined as a mutable scope, everything in it will
    look like "sub-search-space" in the scope. Scopes can be nested.

    There are two ways mutators can use mutable scope. One is to traverse the search space as a tree during initialization
    and reset. The other is to implement `enter_mutable_scope` and `exit_mutable_scope`. They are called before and after
    the forward method of the class inheriting mutable scope.

    Mutable scopes are also mutables that are listed in the mutator.mutables (search space), but they are not supposed
    to appear in the dict of choices.

    Parameters
    ----------
    key : str
        Key of mutable scope.
    """
    def __init__(self, key):
        super().__init__(key=key)

    def _check_built(self):
        return True  # bypass the test because it's deprecated

    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'mutator'):
            return super().__call__(*args, **kwargs)
        warnings.warn("`MutableScope` is deprecated in Retiarii.", DeprecationWarning)
        try:
            self._check_built()
            self.mutator.enter_mutable_scope(self)
            return super().__call__(*args, **kwargs)
        finally:
            self.mutator.exit_mutable_scope(self)


class LayerChoice(Mutable):
    """
    Layer choice selects one of the ``op_candidates``, then apply it on inputs and return results.
    In rare cases, it can also select zero or many.

    Layer choice does not allow itself to be nested.

    Parameters
    ----------
    op_candidates : list of nn.Module or OrderedDict
        A module list to be selected from.
    reduction : str
        ``mean``, ``concat``, ``sum`` or ``none``. Policy if multiples are selected.
        If ``none``, a list is returned. ``mean`` returns the average. ``sum`` returns the sum.
        ``concat`` concatenate the list at dimension 1.
    return_mask : bool
        If ``return_mask``, return output tensor and a mask. Otherwise return tensor only.
    key : str
        Key of the input choice.

    Attributes
    ----------
    length : int
        Deprecated. Number of ops to choose from. ``len(layer_choice)`` is recommended.
    names : list of str
        Names of candidates.
    choices : list of Module
        Deprecated. A list of all candidate modules in the layer choice module.
        ``list(layer_choice)`` is recommended, which will serve the same purpose.

    Notes
    -----
    ``op_candidates`` can be a list of modules or a ordered dict of named modules, for example,

    .. code-block:: python

        self.op_choice = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(3, 16, 128)),
            ("conv5x5", nn.Conv2d(5, 16, 128)),
            ("conv7x7", nn.Conv2d(7, 16, 128))
        ]))

    Elements in layer choice can be modified or deleted. Use ``del self.op_choice["conv5x5"]`` or
    ``self.op_choice[1] = nn.Conv3d(...)``. Adding more choices is not supported yet.
    """

    def __init__(self, op_candidates, reduction="sum", return_mask=False, key=None):
        super().__init__(key=key)
        self.names = []
        if isinstance(op_candidates, OrderedDict):
            for name, module in op_candidates.items():
                assert name not in ["length", "reduction", "return_mask", "_key", "key", "names"], \
                    "Please don't use a reserved name '{}' for your module.".format(name)
                self.add_module(name, module)
                self.names.append(name)
        elif isinstance(op_candidates, list):
            for i, module in enumerate(op_candidates):
                self.add_module(str(i), module)
                self.names.append(str(i))
        else:
            raise TypeError("Unsupported op_candidates type: {}".format(type(op_candidates)))
        self.reduction = reduction
        self.return_mask = return_mask

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._modules[idx]
        return list(self)[idx]

    def __setitem__(self, idx, module):
        key = idx if isinstance(idx, str) else self.names[idx]
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.names[idx]:
                delattr(self, key)
        else:
            if isinstance(idx, str):
                key, idx = idx, self.names.index(idx)
            else:
                key = self.names[idx]
            delattr(self, key)
        del self.names[idx]

    @property
    def length(self):
        warnings.warn("layer_choice.length is deprecated. Use `len(layer_choice)` instead.", DeprecationWarning)
        return len(self)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return map(lambda name: self._modules[name], self.names)

    @property
    def choices(self):
        warnings.warn("layer_choice.choices is deprecated. Use `list(layer_choice)` instead.", DeprecationWarning)
        return list(self)

    def forward(self, *args, **kwargs):
        """
        Returns
        -------
        tuple of tensors
            Output and selection mask. If ``return_mask`` is ``False``, only output is returned.
        """
        out, mask = self.mutator.on_forward_layer_choice(self, *args, **kwargs)
        if self.return_mask:
            return out, mask
        return out


class InputChoice(Mutable):
    """
    Input choice selects ``n_chosen`` inputs from ``choose_from`` (contains ``n_candidates`` keys). For beginners,
    use ``n_candidates`` instead of ``choose_from`` is a safe option. To get the most power out of it, you might want to
    know about ``choose_from``.

    The keys in ``choose_from`` can be keys that appear in past mutables, or ``NO_KEY`` if there are no suitable ones.
    The keys are designed to be the keys of the sources. To help mutators make better decisions,
    mutators might be interested in how the tensors to choose from come into place. For example, the tensor is the
    output of some operator, some node, some cell, or some module. If this operator happens to be a mutable (e.g.,
    ``LayerChoice`` or ``InputChoice``), it has a key naturally that can be used as a source key. If it's a
    module/submodule, it needs to be annotated with a key: that's where a :class:`MutableScope` is needed.

    In the example below, ``input_choice`` is a 4-choose-any. The first 3 is semantically output of cell1, output of cell2,
    output of cell3 with respectively. Notice that an extra max pooling is followed by cell1, indicating x1 is not
    "actually" the direct output of cell1.

    .. code-block:: python

        class Cell(MutableScope):
            pass

        class Net(nn.Module):
            def __init__(self):
                self.cell1 = Cell("cell1")
                self.cell2 = Cell("cell2")
                self.op = LayerChoice([conv3x3(), conv5x5()], key="op")
                self.input_choice = InputChoice(choose_from=["cell1", "cell2", "op", InputChoice.NO_KEY])

            def forward(self, x):
                x1 = max_pooling(self.cell1(x))
                x2 = self.cell2(x)
                x3 = self.op(x)
                x4 = torch.zeros_like(x)
                return self.input_choice([x1, x2, x3, x4])

    Parameters
    ----------
    n_candidates : int
        Number of inputs to choose from.
    choose_from : list of str
        List of source keys to choose from. At least of one of ``choose_from`` and ``n_candidates`` must be fulfilled.
        If ``n_candidates`` has a value but ``choose_from`` is None, it will be automatically treated as ``n_candidates``
        number of empty string.
    n_chosen : int
        Recommended inputs to choose. If None, mutator is instructed to select any.
    reduction : str
        ``mean``, ``concat``, ``sum`` or ``none``. See :class:`LayerChoice`.
    return_mask : bool
        If ``return_mask``, return output tensor and a mask. Otherwise return tensor only.
    key : str
        Key of the input choice.
    """

    NO_KEY = ""

    def __init__(self, n_candidates=None, choose_from=None, n_chosen=None,
                 reduction="sum", return_mask=False, key=None):
        super().__init__(key=key)
        # precondition check
        assert n_candidates is not None or choose_from is not None, "At least one of `n_candidates` and `choose_from`" \
                                                                    "must be not None."
        if choose_from is not None and n_candidates is None:
            n_candidates = len(choose_from)
        elif choose_from is None and n_candidates is not None:
            choose_from = [self.NO_KEY] * n_candidates
        assert n_candidates == len(choose_from), "Number of candidates must be equal to the length of `choose_from`."
        assert n_candidates > 0, "Number of candidates must be greater than 0."
        assert n_chosen is None or 0 <= n_chosen <= n_candidates, "Expected selected number must be None or no more " \
                                                                  "than number of candidates."

        self.n_candidates = n_candidates
        self.choose_from = choose_from.copy()
        self.n_chosen = n_chosen
        self.reduction = reduction
        self.return_mask = return_mask

    def forward(self, optional_inputs):
        """
        Forward method of LayerChoice.

        Parameters
        ----------
        optional_inputs : list or dict
            Recommended to be a dict. As a dict, inputs will be converted to a list that follows the order of
            ``choose_from`` in initialization. As a list, inputs must follow the semantic order that is the same as
            ``choose_from``.

        Returns
        -------
        tuple of tensors
            Output and selection mask. If ``return_mask`` is ``False``, only output is returned.
        """
        optional_input_list = optional_inputs
        if isinstance(optional_inputs, dict):
            optional_input_list = [optional_inputs[tag] for tag in self.choose_from]
        assert isinstance(optional_input_list, list), \
            "Optional input list must be a list, not a {}.".format(type(optional_input_list))
        assert len(optional_inputs) == self.n_candidates, \
            "Length of the input list must be equal to number of candidates."
        out, mask = self.mutator.on_forward_input_choice(self, optional_input_list)
        if self.return_mask:
            return out, mask
        return out
