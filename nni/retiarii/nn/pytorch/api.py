from collections import OrderedDict
from typing import Any, List, Union, Dict
import warnings

import torch
import torch.nn as nn

from ...utils import uid, add_record, del_record


__all__ = ['LayerChoice', 'InputChoice', 'ValueChoice', 'Placeholder', 'ChosenInputs']


class LayerChoice(nn.Module):
    """
    Layer choice selects one of the ``candidates``, then apply it on inputs and return results.

    Layer choice does not allow itself to be nested.

    Parameters
    ----------
    candidates : list of nn.Module or OrderedDict
        A module list to be selected from.
    label : str
        Identifier of the layer choice.

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
    ``candidates`` can be a list of modules or a ordered dict of named modules, for example,

    .. code-block:: python

        self.op_choice = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(3, 16, 128)),
            ("conv5x5", nn.Conv2d(5, 16, 128)),
            ("conv7x7", nn.Conv2d(7, 16, 128))
        ]))

    Elements in layer choice can be modified or deleted. Use ``del self.op_choice["conv5x5"]`` or
    ``self.op_choice[1] = nn.Conv3d(...)``. Adding more choices is not supported yet.
    """

    def __init__(self, candidates: Union[Dict[str, nn.Module], List[nn.Module]], label: str = None, **kwargs):
        super(LayerChoice, self).__init__()
        if 'key' in kwargs:
            warnings.warn(f'"key" is deprecated. Assuming label.')
            label = kwargs['key']
        if 'return_mask' in kwargs:
            warnings.warn(f'"return_mask" is deprecated. Ignoring...')
        if 'reduction' in kwargs:
            warnings.warn(f'"reduction" is deprecated. Ignoring...')
        self.candidates = candidates
        self._label = label if label is not None else f'layerchoice_{uid()}'

        self.names = []
        if isinstance(candidates, OrderedDict):
            for name, module in candidates.items():
                assert name not in ["length", "reduction", "return_mask", "_key", "key", "names"], \
                    "Please don't use a reserved name '{}' for your module.".format(name)
                self.add_module(name, module)
                self.names.append(name)
        elif isinstance(candidates, list):
            for i, module in enumerate(candidates):
                self.add_module(str(i), module)
                self.names.append(str(i))
        else:
            raise TypeError("Unsupported candidates type: {}".format(type(candidates)))

    @property
    def key(self):
        warnings.warn('Using key to access the identifier of LayerChoice is deprecated. Please use label instead.')
        return self._label

    @property
    def label(self):
        return self._label

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

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return map(lambda name: self._modules[name], self.names)

    @property
    def choices(self):
        warnings.warn("layer_choice.choices is deprecated. Use `list(layer_choice)` instead.", DeprecationWarning)
        return list(self)

    def forward(self, x):
        # TODO: add warning when executing in standalone
        return x


class InputChoice(nn.Module):
    """
    Input choice selects ``n_chosen`` inputs from ``choose_from`` (contains ``n_candidates`` keys).
    Use ``reduction`` to specify how chosen inputs are reduced into one output. A few options are:

    * ``none``: do nothing and return the list directly.
    * ``sum``: summing all the chosen inputs.
    * ``mean``: taking the average of all chosen inputs.
    * ``concat``: concatenate all chosen inputs at dimension 1.

    We don't support customizing reduction yet.

    Parameters
    ----------
    n_candidates : int
        Number of inputs to choose from. It is required.
    n_chosen : int
        Recommended inputs to choose. If None, mutator is instructed to select any.
    reduction : str
        ``mean``, ``concat``, ``sum`` or ``none``. See :class:`LayerChoice`.
    label : str
        Identifier of the input choice.
    """

    def __init__(self, n_candidates: int, n_chosen: int = 1, reduction: str = 'sum', label: str = None, **kwargs):
        super(InputChoice, self).__init__()
        if 'key' in kwargs:
            warnings.warn(f'"key" is deprecated. Assuming label.')
            label = kwargs['key']
        if 'return_mask' in kwargs:
            warnings.warn(f'"return_mask" is deprecated. Ignoring...')
        if 'choose_from' in kwargs:
            warnings.warn(f'"reduction" is deprecated. Ignoring...')
        self.n_candidates = n_candidates
        self.n_chosen = n_chosen
        self.reduction = reduction
        self._label = label if label is not None else f'inputchoice_{uid()}'

    @property
    def key(self):
        warnings.warn('Using key to access the identifier of InputChoice is deprecated. Please use label instead.')
        return self._label

    @property
    def label(self):
        return self._label

    def forward(self, candidate_inputs: List[torch.Tensor]) -> torch.Tensor:
        # TODO: add warning when executing in standalone
        return candidate_inputs[0]


class ValueChoice(nn.Module):
    """
    ValueChoice is to choose one from ``candidates``.

    Should initialize the values to choose from in init and call the module in forward to get the chosen value.

    Parameters
    ----------
    candidates : list
        List of values to choose from.
    """

    def __init__(self, candidates: List[Any]):
        self.candidates = candidates

    def forward(self):
        # TODO: add warning when executing in standalone
        return 0


class Placeholder(nn.Module):
    # docstring: TODO

    def __init__(self, label, related_info):
        add_record(id(self), related_info)
        self.label = label
        self.related_info = related_info
        super(Placeholder, self).__init__()

    def forward(self, x):
        return x

    def __del__(self):
        del_record(id(self))


class ChosenInputs(nn.Module):
    """
    A module that chooses from a tensor list and outputs a reduced tensor.
    The already-chosen version of InputChoice.
    """

    def __init__(self, chosen: List[int], reduction: str):
        super().__init__()
        self.chosen = chosen
        self.reduction = reduction

    def forward(self, candidate_inputs):
        return self._tensor_reduction(self.reduction, [candidate_inputs[i] for i in self.chosen])

    def _tensor_reduction(self, reduction_type, tensor_list):
        if reduction_type == 'none':
            return tensor_list
        if not tensor_list:
            return None  # empty. return None for now
        if len(tensor_list) == 1:
            return tensor_list[0]
        if reduction_type == 'sum':
            return sum(tensor_list)
        if reduction_type == 'mean':
            return sum(tensor_list) / len(tensor_list)
        if reduction_type == 'concat':
            return torch.cat(tensor_list, dim=1)
        raise ValueError(f'Unrecognized reduction policy: "{reduction_type}"')
