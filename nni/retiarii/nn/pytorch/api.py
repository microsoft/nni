# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import operator
import warnings
from typing import Any, List, Union, Dict, Optional, Callable, Iterable, NoReturn, TypeVar

import torch
import torch.nn as nn

from nni.common.serializer import Translatable
from nni.retiarii.serializer import basic_unit
from nni.retiarii.utils import STATE_DICT_PY_MAPPING_PARTIAL
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
            result = candidates[int(chosen)]
        else:
            result = candidates[chosen]

        # map the named hierarchies to support weight inheritance for python engine
        if hasattr(result, STATE_DICT_PY_MAPPING_PARTIAL):
            # handle cases where layer choices are nested
            # already has a mapping, will merge with it
            prev_mapping = getattr(result, STATE_DICT_PY_MAPPING_PARTIAL)
            setattr(result, STATE_DICT_PY_MAPPING_PARTIAL, {k: f'{chosen}.{v}' for k, v in prev_mapping.items()})
        else:
            # "result" needs to know where to map itself.
            # Ideally, we should put a _mapping_ in the module where "result" is located,
            # but it's impossible to put mapping into parent module here.
            setattr(result, STATE_DICT_PY_MAPPING_PARTIAL, {'__self__': str(chosen)})
        return result

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


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

ReductionType = Literal['mean', 'concat', 'sum', 'none']


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
    def create_fixed_module(cls, n_candidates: int, n_chosen: Optional[int] = 1,
                            reduction: ReductionType = 'sum', *,
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


class ChosenInputs(nn.Module):
    """
    A module that chooses from a tensor list and outputs a reduced tensor.
    The already-chosen version of InputChoice.

    When forward, ``chosen`` will be used to select inputs from ``candidate_inputs``,
    and ``reduction`` will be used to choose from those inputs to form a tensor.

    Attributes
    ----------
    chosen : list of int
        Indices of chosen inputs.
    reduction : ``mean`` | ``concat`` | ``sum`` | ``none``
        How to reduce the inputs when multiple are selected.
    """

    def __init__(self, chosen: Union[List[int], int], reduction: ReductionType):
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


# the code in ValueChoice can be generated with this codegen
# this is not done online because I want to have type-hint supports
# $ python -c "from nni.retiarii.nn.pytorch.api import _valuechoice_codegen; _valuechoice_codegen(_internal=True)"
def _valuechoice_codegen(*, _internal: bool = False):
    if not _internal:
        raise RuntimeError("This method is set to be internal. Please don't use it directly.")
    MAPPING = {
        # unary
        'neg': '-', 'pos': '+', 'invert': '~',
        # binary
        'add': '+', 'sub': '-', 'mul': '*', 'matmul': '@',
        'truediv': '//', 'floordiv': '/', 'mod': '%',
        'lshift': '<<', 'rshift': '>>',
        'and': '&', 'xor': '^', 'or': '|',
        # no reflection
        'lt': '<', 'le': '<=', 'eq': '==',
        'ne': '!=', 'ge': '>=', 'gt': '>',
        # NOTE
        # Currently we don't support operators like __contains__ (b in a),
        # Might support them in future when we actually need them.
    }

    binary_template = """    def __{op}__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.{opt}, '{{}} {sym} {{}}', [self, other])"""

    binary_r_template = """    def __r{op}__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.{opt}, '{{}} {sym} {{}}', [other, self])"""

    unary_template = """    def __{op}__(self) -> 'ValueChoiceX':
        return ValueChoiceX(operator.{op}, '{sym}{{}}', [self])"""

    for op, sym in MAPPING.items():
        if op in ['neg', 'pos', 'invert']:
            print(unary_template.format(op=op, sym=sym) + '\n')
        else:
            opt = op + '_' if op in ['and', 'or'] else op
            print(binary_template.format(op=op, opt=opt, sym=sym) + '\n')
            if op not in ['lt', 'le', 'eq', 'ne', 'ge', 'gt']:
                print(binary_r_template.format(op=op, opt=opt, sym=sym) + '\n')


def _valuechoice_staticmethod_helper(orig_func):
    orig_func.__doc__ += """
        Notes
        -----
        This function performs lazy evaluation.
        Only the expression will be recorded when the function is called.
        The real evaluation happens when the inner value choice has determined its final decision.
        If no value choice is contained in the parameter list, the evaluation will be intermediate."""
    return orig_func


class ValueChoiceX(Translatable):
    """Internal API. Implementation note:

    The transformed (X) version of value choice.
    It can be the result of composition (transformation) of one or several value choices. For example,

    .. code-block:: python

        nn.ValueChoice([1, 2]) + nn.ValueChoice([3, 4]) + 5

    The instance of base class cannot be created directly. Instead, they should be only the result of transformation of value choice.
    Therefore, there is no need to implement ``create_fixed_module`` in this class, because,
    1. For python-engine, value choice itself has create fixed module. Consequently, the transformation is born to be fixed.
    2. For graph-engine, it uses evaluate to calculate the result.

    Potentially, we have to implement the evaluation logic in oneshot algorithms. I believe we can postpone the discussion till then.
    """

    def __init__(self, function: Callable[..., Any], repr_template: str, arguments: List[Any], dry_run: bool = True):
        super().__init__()

        if function is None:
            # this case is a hack for ValueChoice subclass
            # it will reach here only because ``__init__`` in ``nn.Module`` is useful.
            return

        self.function = function
        self.repr_template = repr_template
        self.arguments = arguments

        assert any(isinstance(arg, ValueChoiceX) for arg in self.arguments)

        if dry_run:
            # for sanity check
            self.dry_run()

    def inner_choices(self) -> Iterable['ValueChoice']:
        """
        Return an iterable of all leaf value choices.
        Useful for composition of value choices.
        No deduplication on labels. Mutators should take care.
        """
        for arg in self.arguments:
            if isinstance(arg, ValueChoiceX):
                yield from arg.inner_choices()

    def dry_run(self) -> Any:
        """
        Dry run the value choice to get one of its possible evaluation results.
        """
        # values are not used
        return self._evaluate(iter([]), True)

    def evaluate(self, values: Iterable[Any]) -> Any:
        """
        Evaluate the result of this group.
        ``values`` should in the same order of ``inner_choices()``.
        """
        return self._evaluate(iter(values), False)

    def _evaluate(self, values: Iterable[Any], dry_run: bool = False) -> Any:
        # "values" iterates in the recursion
        eval_args = []
        for arg in self.arguments:
            if isinstance(arg, ValueChoiceX):
                # recursive evaluation
                eval_args.append(arg._evaluate(values, dry_run))
                # the recursion will stop when it hits a leaf node (value choice)
                # the implementation is in `ValueChoice`
            else:
                # constant value
                eval_args.append(arg)
        return self.function(*eval_args)

    def _translate(self):
        """
        Try to behave like one of its candidates when used in ``basic_unit``.
        """
        return self.dry_run()

    def __repr__(self):
        reprs = []
        for arg in self.arguments:
            if isinstance(arg, ValueChoiceX) and not isinstance(arg, ValueChoice):
                reprs.append('(' + repr(arg) + ')')  # add parenthesis for operator priority
            else:
                reprs.append(repr(arg))
        return self.repr_template.format(*reprs)

    # the following are a series of methods to create "ValueChoiceX"
    # which is a transformed version of value choice
    # https://docs.python.org/3/reference/datamodel.html#special-method-names

    # Special operators that can be useful in place of built-in conditional operators.
    @staticmethod
    @_valuechoice_staticmethod_helper
    def to_int(obj: 'ValueChoiceOrAny') -> Union['ValueChoiceX', int]:
        """
        Convert a ``ValueChoice`` to an integer.
        """
        if isinstance(obj, ValueChoiceX):
            return ValueChoiceX(int, 'int({})', [obj])
        return int(obj)

    @staticmethod
    @_valuechoice_staticmethod_helper
    def to_float(obj: 'ValueChoiceOrAny') -> Union['ValueChoiceX', float]:
        """
        Convert a ``ValueChoice`` to a float.
        """
        if isinstance(obj, ValueChoiceX):
            return ValueChoiceX(float, 'float({})', [obj])
        return float(obj)

    @staticmethod
    @_valuechoice_staticmethod_helper
    def condition(pred: 'ValueChoiceOrAny',
                  true: 'ValueChoiceOrAny',
                  false: 'ValueChoiceOrAny') -> 'ValueChoiceOrAny':
        """
        Return ``true`` if the predicate ``pred`` is true else ``false``.

        Examples
        --------
        >>> ValueChoice.condition(ValueChoice([1, 2]) > ValueChoice([0, 3]), 2, 1)
        """
        if any(isinstance(obj, ValueChoiceX) for obj in [pred, true, false]):
            return ValueChoiceX(lambda t, c, f: t if c else f, '{} if {} else {}', [true, pred, false])
        return true if pred else false

    @staticmethod
    @_valuechoice_staticmethod_helper
    def max(arg0: Union[Iterable['ValueChoiceOrAny'], 'ValueChoiceOrAny'],
            *args: List['ValueChoiceOrAny']) -> 'ValueChoiceOrAny':
        """
        Returns the maximum value from a list of value choices.
        The usage should be similar to Python's built-in value choices,
        where the parameters could be an iterable, or at least two arguments.
        """
        if not args:
            return ValueChoiceX.max(*list(arg0))
        lst = [arg0] + list(args)
        if any(isinstance(obj, ValueChoiceX) for obj in lst):
            return ValueChoiceX(max, 'max({})', lst)
        return max(lst)

    @staticmethod
    @_valuechoice_staticmethod_helper
    def min(arg0: Union[Iterable['ValueChoiceOrAny'], 'ValueChoiceOrAny'],
            *args: List['ValueChoiceOrAny']) -> 'ValueChoiceOrAny':
        """
        Returns the minunum value from a list of value choices.
        The usage should be similar to Python's built-in value choices,
        where the parameters could be an iterable, or at least two arguments.
        """
        if not args:
            return ValueChoiceX.min(*list(arg0))
        lst = [arg0] + list(args)
        if any(isinstance(obj, ValueChoiceX) for obj in lst):
            return ValueChoiceX(min, 'min({})', lst)
        return min(lst)

    def __hash__(self):
        # this is required because we have implemented ``__eq__``
        return id(self)

    # NOTE:
    # Write operations are not supported. Reasons follow:
    # - Semantics are not clear. It can be applied to "all" the inner candidates, or only the chosen one.
    # - Implementation effort is too huge.
    # As a result, inplace operators like +=, *=, magic methods like `__getattr__` are not included in this list.

    def __getitem__(self, key: Any) -> 'ValueChoiceX':
        return ValueChoiceX(lambda x, y: x[y], '{}[{}]', [self, key])

    # region implement int, float, round, trunc, floor, ceil
    # because I believe sometimes we need them to calculate #channels
    # `__int__` and `__float__` are not supported because `__int__` is required to return int.
    def __round__(self, ndigits: Optional[Any] = None) -> 'ValueChoiceX':
        if ndigits is not None:
            return ValueChoiceX(round, 'round({}, {})', [self, ndigits])
        return ValueChoiceX(round, 'round({})', [self])

    def __trunc__(self) -> 'ValueChoiceX':
        raise RuntimeError("Try to use `ValueChoice.to_int()` instead of `math.trunc()` on value choices.")

    def __floor__(self) -> 'ValueChoiceX':
        return ValueChoiceX(math.floor, 'math.floor({})', [self])

    def __ceil__(self) -> 'ValueChoiceX':
        return ValueChoiceX(math.ceil, 'math.ceil({})', [self])

    def __index__(self) -> NoReturn:
        # https://docs.python.org/3/reference/datamodel.html#object.__index__
        raise RuntimeError("`__index__` is not allowed on ValueChoice, which means you can't "
                           "use int(), float(), complex(), range() on a ValueChoice.")

    def __bool__(self) -> NoReturn:
        raise RuntimeError('Cannot use bool() on ValueChoice. That means, using ValueChoice in a if-clause is illegal. '
                           'Please try methods like `ValueChoice.max(a, b)` to see whether that meets your needs.')
    # endregion

    # region the following code is generated with codegen (see above)
    # Annotated with "region" because I want to collapse them in vscode
    def __neg__(self) -> 'ValueChoiceX':
        return ValueChoiceX(operator.neg, '-{}', [self])

    def __pos__(self) -> 'ValueChoiceX':
        return ValueChoiceX(operator.pos, '+{}', [self])

    def __invert__(self) -> 'ValueChoiceX':
        return ValueChoiceX(operator.invert, '~{}', [self])

    def __add__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.add, '{} + {}', [self, other])

    def __radd__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.add, '{} + {}', [other, self])

    def __sub__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.sub, '{} - {}', [self, other])

    def __rsub__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.sub, '{} - {}', [other, self])

    def __mul__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.mul, '{} * {}', [self, other])

    def __rmul__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.mul, '{} * {}', [other, self])

    def __matmul__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.matmul, '{} @ {}', [self, other])

    def __rmatmul__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.matmul, '{} @ {}', [other, self])

    def __truediv__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.truediv, '{} // {}', [self, other])

    def __rtruediv__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.truediv, '{} // {}', [other, self])

    def __floordiv__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.floordiv, '{} / {}', [self, other])

    def __rfloordiv__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.floordiv, '{} / {}', [other, self])

    def __mod__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.mod, '{} % {}', [self, other])

    def __rmod__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.mod, '{} % {}', [other, self])

    def __lshift__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.lshift, '{} << {}', [self, other])

    def __rlshift__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.lshift, '{} << {}', [other, self])

    def __rshift__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.rshift, '{} >> {}', [self, other])

    def __rrshift__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.rshift, '{} >> {}', [other, self])

    def __and__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.and_, '{} & {}', [self, other])

    def __rand__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.and_, '{} & {}', [other, self])

    def __xor__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.xor, '{} ^ {}', [self, other])

    def __rxor__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.xor, '{} ^ {}', [other, self])

    def __or__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.or_, '{} | {}', [self, other])

    def __ror__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.or_, '{} | {}', [other, self])

    def __lt__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.lt, '{} < {}', [self, other])

    def __le__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.le, '{} <= {}', [self, other])

    def __eq__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.eq, '{} == {}', [self, other])

    def __ne__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.ne, '{} != {}', [self, other])

    def __ge__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.ge, '{} >= {}', [self, other])

    def __gt__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(operator.gt, '{} > {}', [self, other])
    # endregion

    # __pow__, __divmod__, __abs__ are special ones.
    # Not easy to cover those cases with codegen.
    def __pow__(self, other: Any, modulo: Optional[Any] = None) -> 'ValueChoiceX':
        if modulo is not None:
            return ValueChoiceX(pow, 'pow({}, {}, {})', [self, other, modulo])
        return ValueChoiceX(lambda a, b: a ** b, '{} ** {}', [self, other])

    def __rpow__(self, other: Any, modulo: Optional[Any] = None) -> 'ValueChoiceX':
        if modulo is not None:
            return ValueChoiceX(pow, 'pow({}, {}, {})', [other, self, modulo])
        return ValueChoiceX(lambda a, b: a ** b, '{} ** {}', [other, self])

    def __divmod__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(divmod, 'divmod({}, {})', [self, other])

    def __rdivmod__(self, other: Any) -> 'ValueChoiceX':
        return ValueChoiceX(divmod, 'divmod({}, {})', [other, self])

    def __abs__(self) -> 'ValueChoiceX':
        return ValueChoiceX(abs, 'abs({})', [self])


ValueChoiceOrAny = TypeVar('ValueChoiceOrAny', ValueChoiceX, Any)


class ValueChoice(ValueChoiceX, Mutable):
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
        value = get_fixed_value(label)
        if value not in candidates:
            raise ValueError(f'Value {value} does not belong to the candidates: {candidates}.')
        return value

    def __init__(self, candidates: List[Any], *, prior: Optional[List[float]] = None, label: Optional[str] = None):
        super().__init__(None, None, None)
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

    def inner_choices(self) -> Iterable['ValueChoice']:
        # yield self because self is the only value choice here
        yield self

    def dry_run(self) -> Any:
        return self.candidates[0]

    def _evaluate(self, values: Iterable[Any], dry_run: bool = False) -> Any:
        if dry_run:
            return self.candidates[0]
        try:
            value = next(values)
        except StopIteration:
            raise ValueError(f'Value list {values} is exhausted when trying to get a chosen value of {self}.')
        if value not in self.candidates:
            raise ValueError(f'Value {value} does not belong to the candidates of {self}.')
        return value

    def __repr__(self):
        return f'ValueChoice({self.candidates}, label={repr(self.label)})'


@basic_unit
class Placeholder(nn.Module):
    """
    The API that creates an empty module for later mutations.
    For advanced usages only.
    """

    def __init__(self, label, **related_info):
        self.label = label
        self.related_info = related_info
        super().__init__()

    def forward(self, x):
        return x
