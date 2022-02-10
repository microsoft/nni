# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import warnings
from typing import Any, List, Union, Dict, Optional

import torch
import torch.nn as nn

from nni.common.serializer import Translatable
from nni.retiarii.serializer import basic_unit
from .utils import Mutable, generate_new_label, get_fixed_value


__all__ = ['LayerChoice', 'InputChoice', 'ValueChoice', 'Placeholder', 'ChosenInputs']


class LayerChoice(Mutable):
    """
    Layer choice selects one of the ``candidates``, then apply it on inputs and return results.

    Layer choice does not allow itself to be nested.

    Parameters
    ----------
    candidates : list of nn.Module or OrderedDict
        A module list to be selected from.
    prior : list of float
        Prior distribution used in random sampling.
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

    # FIXME: prior is designed but not supported yet

    @classmethod
    def create_fixed_module(cls, candidates: Union[Dict[str, nn.Module], List[nn.Module]], *,
                            label: Optional[str] = None, **kwargs):
        chosen = get_fixed_value(label)
        if isinstance(candidates, list):
            return candidates[int(chosen)]
        else:
            return candidates[chosen]

    def __init__(self, candidates: Union[Dict[str, nn.Module], List[nn.Module]], *,
                 prior: Optional[List[float]] = None, label: Optional[str] = None, **kwargs):
        super(LayerChoice, self).__init__()
        if 'key' in kwargs:
            warnings.warn(f'"key" is deprecated. Assuming label.')
            label = kwargs['key']
        if 'return_mask' in kwargs:
            warnings.warn(f'"return_mask" is deprecated. Ignoring...')
        if 'reduction' in kwargs:
            warnings.warn(f'"reduction" is deprecated. Ignoring...')
        self.candidates = candidates
        self.prior = prior or [1 / len(candidates) for _ in range(len(candidates))]
        assert abs(sum(self.prior) - 1) < 1e-5, 'Sum of prior distribution is not 1.'
        self._label = generate_new_label(label)

        self.names = []
        if isinstance(candidates, dict):
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
        self._first_module = self._modules[self.names[0]]  # to make the dummy forward meaningful

    @property
    def key(self):
        return self._key()

    @torch.jit.ignore
    def _key(self):
        warnings.warn('Using key to access the identifier of LayerChoice is deprecated. Please use label instead.',
                      category=DeprecationWarning)
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
        return self._choices()

    @torch.jit.ignore
    def _choices(self):
        warnings.warn("layer_choice.choices is deprecated. Use `list(layer_choice)` instead.", category=DeprecationWarning)
        return list(self)

    def forward(self, x):
        warnings.warn('You should not run forward of this module directly.')
        return self._first_module(x)

    def __repr__(self):
        return f'LayerChoice({self.candidates}, label={repr(self.label)})'


class InputChoice(Mutable):
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
        ``mean``, ``concat``, ``sum`` or ``none``.
    prior : list of float
        Prior distribution used in random sampling.
    label : str
        Identifier of the input choice.
    """

    @classmethod
    def create_fixed_module(cls, n_candidates: int, n_chosen: Optional[int] = 1, reduction: str = 'sum', *,
                            prior: Optional[List[float]] = None, label: Optional[str] = None, **kwargs):
        return ChosenInputs(get_fixed_value(label), reduction=reduction)

    def __init__(self, n_candidates: int, n_chosen: Optional[int] = 1,
                 reduction: str = 'sum', *,
                 prior: Optional[List[float]] = None, label: Optional[str] = None, **kwargs):
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
        self.prior = prior or [1 / n_candidates for _ in range(n_candidates)]
        assert self.reduction in ['mean', 'concat', 'sum', 'none']
        self._label = generate_new_label(label)

    @property
    def key(self):
        return self._key()

    @torch.jit.ignore
    def _key(self):
        warnings.warn('Using key to access the identifier of InputChoice is deprecated. Please use label instead.',
                      category=DeprecationWarning)
        return self._label

    @property
    def label(self):
        return self._label

    def forward(self, candidate_inputs: List[torch.Tensor]) -> torch.Tensor:
        warnings.warn('You should not run forward of this module directly.')
        return candidate_inputs[0]

    def __repr__(self):
        return f'InputChoice(n_candidates={self.n_candidates}, n_chosen={self.n_chosen}, ' \
            f'reduction={repr(self.reduction)}, label={repr(self.label)})'


class ValueChoice(Translatable, Mutable):
    """
    ValueChoice is to choose one from ``candidates``.

    In most use scenarios, ValueChoice should be passed to the init parameters of a serializable module. For example,

    .. code-block:: python

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, nn.ValueChoice([32, 64]), kernel_size=nn.ValueChoice([3, 5, 7]))

            def forward(self, x):
                return self.conv(x)

    In case, you want to search a parameter that is used repeatedly, this is also possible by sharing the same value choice instance.
    (Sharing the label should have the same effect.) For example,

    .. code-block:: python

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                hidden_dim = nn.ValueChoice([128, 512])
                self.fc = nn.Sequential(
                    nn.Linear(64, hidden_dim),
                    nn.Linear(hidden_dim, 10)
                )

                # the following code has the same effect.
                # self.fc = nn.Sequential(
                #     nn.Linear(64, nn.ValueChoice([128, 512], label='dim')),
                #     nn.Linear(nn.ValueChoice([128, 512], label='dim'), 10)
                # )

            def forward(self, x):
                return self.fc(x)

    Note that ValueChoice should be used directly. Transformations like ``nn.Linear(32, nn.ValueChoice([64, 128]) * 2)``
    are not supported.

    Another common use case is to initialize the values to choose from in init and call the module in forward to get the chosen value.
    Usually, this is used to pass a mutable value to a functional API like ``torch.xxx`` or ``nn.functional.xxx```.
    For example,

    .. code-block:: python

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout_rate = nn.ValueChoice([0., 1.])

            def forward(self, x):
                return F.dropout(x, self.dropout_rate())

    Parameters
    ----------
    candidates : list
        List of values to choose from.
    prior : list of float
        Prior distribution to sample from.
    label : str
        Identifier of the value choice.
    """

    # FIXME: prior is designed but not supported yet

    @classmethod
    def create_fixed_module(cls, candidates: List[Any], *, label: Optional[str] = None, **kwargs):
        return get_fixed_value(label)

    def __init__(self, candidates: List[Any], *, prior: Optional[List[float]] = None, label: Optional[str] = None):
        super().__init__()
        self.candidates = candidates
        self.prior = prior or [1 / len(candidates) for _ in range(len(candidates))]
        assert abs(sum(self.prior) - 1) < 1e-5, 'Sum of prior distribution is not 1.'
        self._label = generate_new_label(label)
        self._accessor = []

    @property
    def label(self):
        return self._label

    def forward(self):
        warnings.warn('You should not run forward of this module directly.')
        return self.candidates[0]

    def _translate(self):
        # Will function as a value when used in serializer.
        return self.access(self.candidates[0])

    def __repr__(self):
        return f'ValueChoice({self.candidates}, label={repr(self.label)})'

    def access(self, value):
        if not self._accessor:
            return value
        try:
            v = value
            for a in self._accessor:
                v = v[a]
        except KeyError:
            raise KeyError(''.join([f'[{a}]' for a in self._accessor]) + f' does not work on {value}')
        return v

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        new_item = ValueChoice(self.candidates, label=self.label)
        new_item._accessor = [*self._accessor]
        return new_item

    def __getitem__(self, item):
        """
        Get a sub-element of value choice.

        The underlying implementation is to clone the current instance, and append item to "accessor", which records all
        the history getitem calls. For example, when accessor is ``[a, b, c]``, the value choice will return ``vc[a][b][c]``
        where ``vc`` is the original value choice.
        """
        access = copy.deepcopy(self)
        access._accessor.append(item)
        for candidate in self.candidates:
            access.access(candidate)
        return access


@basic_unit
class Placeholder(nn.Module):
    # TODO: docstring

    def __init__(self, label, **related_info):
        self.label = label
        self.related_info = related_info
        super().__init__()

    def forward(self, x):
        return x


class ChosenInputs(nn.Module):
    """
    A module that chooses from a tensor list and outputs a reduced tensor.
    The already-chosen version of InputChoice.
    """

    def __init__(self, chosen: Union[List[int], int], reduction: str):
        super().__init__()
        self.chosen = chosen if isinstance(chosen, list) else [chosen]
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
