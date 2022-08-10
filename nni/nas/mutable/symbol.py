# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Infrastructure for symbolic execution.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Iterable, TypeVar, NoReturn

_func = TypeVar('_func')

T = TypeVar('T')

# the code in symbol expression can be generated with this codegen
# this is not done online because I want to have type-hint supports
# $ python -c "from nni.nas.mutable.symbol import _symbol_expr_codegen; _symbol_expr_codegen(_internal=True)"
def _symbol_expr_codegen(*, _internal: bool = False):
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
        # no reverse
        'lt': '<', 'le': '<=', 'eq': '==',
        'ne': '!=', 'ge': '>=', 'gt': '>',
        # NOTE
        # Currently we don't support operators like __contains__ (b in a),
        # Might support them in future when we actually need them.
    }

    binary_template = """    def __{op}__(self: T, other: Any) -> T:
        return self.__class__(operator.{opt}, '{{}} {sym} {{}}', [self, other])"""

    binary_r_template = """    def __r{op}__(self: T, other: Any) -> T:
        return self.__class__(operator.{opt}, '{{}} {sym} {{}}', [other, self])"""

    unary_template = """    def __{op}__(self: T) -> T:
        return self.__class__(operator.{op}, '{sym}{{}}', [self])"""

    for op, sym in MAPPING.items():
        if op in ['neg', 'pos', 'invert']:
            print(unary_template.format(op=op, sym=sym) + '\n')
        else:
            opt = op + '_' if op in ['and', 'or'] else op
            print(binary_template.format(op=op, opt=opt, sym=sym) + '\n')
            if op not in ['lt', 'le', 'eq', 'ne', 'ge', 'gt']:
                print(binary_r_template.format(op=op, opt=opt, sym=sym) + '\n')


def symbolic_staticmethod(orig_func: _func) -> _func:
    if orig_func.__doc__ is not None:
        orig_func.__doc__ += """
        Notes
        -----
        This function performs lazy evaluation.
        Only the expression will be recorded when the function is called.
        The real evaluation happens when the inner value choice has determined its final decision.
        If no value choice is contained in the parameter list, the evaluation will be intermediate."""
    return staticmethod(orig_func)


def first_symbolic_object(*objects: Any) -> SymbolicExpression | None:
    """
    Return the first symbolic object in the given list.
    """
    for obj in objects:
        if isinstance(obj, SymbolicExpression):
            return obj
    return None


class SymbolicExpression:
    """Implementation of symbolic execution.

    Each instance of :class:`SymbolicExpression` is a node on the tree,
    with a function and a list of children (i.e., function arguments).

    The expression is designed to be compatible with native Python expressions.
    That means, the static methods (as well as operators) can be also applied on plain Python values.
    """

    # Python special methods list:
    # https://docs.python.org/3/reference/datamodel.html#special-method-names

    # Special operators that can be useful in place of built-in conditional operators.
    @symbolic_staticmethod
    def to_int(obj: Any) -> SymbolicExpression | int:
        """Convert the current value to an integer."""
        if isinstance(obj, SymbolicExpression):
            return obj.__class__(int, 'int({})', [obj])
        return int(obj)

    @symbolic_staticmethod
    def to_float(obj: Any) -> SymbolicExpression | float:
        """Convert the current value to a float."""
        if isinstance(obj, SymbolicExpression):
            return obj.__class__(float, 'float({})', [obj])
        return float(obj)

    @symbolic_staticmethod
    def condition(pred: Any, true: Any, false: Any) -> SymbolicExpression | Any:
        """
        Return ``true`` if the predicate ``pred`` is true else ``false``.

        Examples
        --------
        >>> SymbolicExpression.condition(Symbol('x') > Symbol('y'), 2, 1)
        """
        symbol_obj = first_symbolic_object(pred, true, false)
        if symbol_obj is not None:
            return symbol_obj.__class__(lambda t, c, f: t if c else f, '{} if {} else {}', [true, pred, false])
        return true if pred else false

    @symbolic_staticmethod
    def max(arg0: Iterable[Any] | Any, *args: Any) -> Any:
        """
        Returns the maximum value from a list of symbols.
        The usage should be similar to Python's built-in symbols,
        where the parameters could be an iterable, or at least two arguments.
        """
        if not args:
            if not isinstance(arg0, Iterable):
                raise TypeError('Expect more than one items to compare max')
            return SymbolicExpression.max(*list(arg0))
        lst = list(arg0) if isinstance(arg0, Iterable) else [arg0] + list(args)
        symbol_obj = first_symbolic_object(*lst)
        if symbol_obj is not None:
            return symbol_obj.__class__(max, 'max({})', lst)
        return max(lst)

    @symbolic_staticmethod
    def min(arg0: Iterable[Any] | Any, *args: Any) -> Any:
        """
        Returns the minimum value from a list of symbols.
        The usage should be similar to Python's built-in symbols,
        where the parameters could be an iterable, or at least two arguments.
        """
        if not args:
            if not isinstance(arg0, Iterable):
                raise TypeError('Expect more than one items to compare min')
            return SymbolicExpression.min(*list(arg0))
        lst = list(arg0) if isinstance(arg0, Iterable) else [arg0] + list(args)
        symbol_obj = first_symbolic_object(*lst)
        if symbol_obj is not None:
            return symbol_obj.__class__(min, 'min({})', lst)
        return min(lst)

    def __hash__(self):
        # this is required because we have implemented ``__eq__``
        return id(self)

    # NOTE:
    # Write operations are not supported. Reasons follow:
    # - Semantics are not clear. It can be applied to "all" the inner candidates, or only the chosen one.
    # - Implementation effort is too huge.
    # As a result, inplace operators like +=, *=, magic methods like `__getattr__` are not included in this list.

    def __getitem__(self: T, key: Any) -> T:
        return self.__class__(lambda x, y: x[y], '{}[{}]', [self, key])

    # region implement int, float, round, trunc, floor, ceil
    # because I believe sometimes we need them to calculate #channels
    # `__int__` and `__float__` are not supported because `__int__` is required to return int.
    def __round__(self: T, ndigits: Any | int | None = None) -> Any:
        if ndigits is not None:
            return self.__class__(round, 'round({}, {})', [self, ndigits])
        return self.__class__(round, 'round({})', [self])

    def __trunc__(self) -> NoReturn:
        raise RuntimeError("Try to use `SymbolicExpression.to_int()` instead of `math.trunc()` on value choices.")

    def __floor__(self: T) -> T:
        return self.__class__(math.floor, 'math.floor({})', [self])

    def __ceil__(self: T) -> T:
        return self.__class__(math.ceil, 'math.ceil({})', [self])

    def __index__(self) -> NoReturn:
        # https://docs.python.org/3/reference/datamodel.html#object.__index__
        raise RuntimeError("`__index__` is not allowed on SymbolicExpression, which means you can't "
                           "use int(), float(), complex(), range() on a SymbolicExpression. "
                           "To cast the type of SymbolicExpression, please try `SymbolicExpression.to_int()` "
                           "or `SymbolicExpression.to_float()`.")

    def __bool__(self) -> NoReturn:
        raise RuntimeError('Cannot use bool() on SymbolicExpression. That means, using SymbolicExpression in a if-clause is illegal. '
                           'Please try methods like `SymbolicExpression.max(a, b)` to see whether that meets your needs.')
    # endregion

    # region the following code is generated with codegen (see above)
    # Annotated with "region" because I want to collapse them in vscode
    def __neg__(self: T) -> T:
        return self.__class__(operator.neg, '-{}', [self])

    def __pos__(self: T) -> T:
        return self.__class__(operator.pos, '+{}', [self])

    def __invert__(self: T) -> T:
        return self.__class__(operator.invert, '~{}', [self])

    def __add__(self: T, other: Any) -> T:
        return self.__class__(operator.add, '{} + {}', [self, other])

    def __radd__(self: T, other: Any) -> T:
        return self.__class__(operator.add, '{} + {}', [other, self])

    def __sub__(self: T, other: Any) -> T:
        return self.__class__(operator.sub, '{} - {}', [self, other])

    def __rsub__(self: T, other: Any) -> T:
        return self.__class__(operator.sub, '{} - {}', [other, self])

    def __mul__(self: T, other: Any) -> T:
        return self.__class__(operator.mul, '{} * {}', [self, other])

    def __rmul__(self: T, other: Any) -> T:
        return self.__class__(operator.mul, '{} * {}', [other, self])

    def __matmul__(self: T, other: Any) -> T:
        return self.__class__(operator.matmul, '{} @ {}', [self, other])

    def __rmatmul__(self: T, other: Any) -> T:
        return self.__class__(operator.matmul, '{} @ {}', [other, self])

    def __truediv__(self: T, other: Any) -> T:
        return self.__class__(operator.truediv, '{} // {}', [self, other])

    def __rtruediv__(self: T, other: Any) -> T:
        return self.__class__(operator.truediv, '{} // {}', [other, self])

    def __floordiv__(self: T, other: Any) -> T:
        return self.__class__(operator.floordiv, '{} / {}', [self, other])

    def __rfloordiv__(self: T, other: Any) -> T:
        return self.__class__(operator.floordiv, '{} / {}', [other, self])

    def __mod__(self: T, other: Any) -> T:
        return self.__class__(operator.mod, '{} % {}', [self, other])

    def __rmod__(self: T, other: Any) -> T:
        return self.__class__(operator.mod, '{} % {}', [other, self])

    def __lshift__(self: T, other: Any) -> T:
        return self.__class__(operator.lshift, '{} << {}', [self, other])

    def __rlshift__(self: T, other: Any) -> T:
        return self.__class__(operator.lshift, '{} << {}', [other, self])

    def __rshift__(self: T, other: Any) -> T:
        return self.__class__(operator.rshift, '{} >> {}', [self, other])

    def __rrshift__(self: T, other: Any) -> T:
        return self.__class__(operator.rshift, '{} >> {}', [other, self])

    def __and__(self: T, other: Any) -> T:
        return self.__class__(operator.and_, '{} & {}', [self, other])

    def __rand__(self: T, other: Any) -> T:
        return self.__class__(operator.and_, '{} & {}', [other, self])

    def __xor__(self: T, other: Any) -> T:
        return self.__class__(operator.xor, '{} ^ {}', [self, other])

    def __rxor__(self: T, other: Any) -> T:
        return self.__class__(operator.xor, '{} ^ {}', [other, self])

    def __or__(self: T, other: Any) -> T:
        return self.__class__(operator.or_, '{} | {}', [self, other])

    def __ror__(self: T, other: Any) -> T:
        return self.__class__(operator.or_, '{} | {}', [other, self])

    def __lt__(self: T, other: Any) -> T:
        return self.__class__(operator.lt, '{} < {}', [self, other])

    def __le__(self: T, other: Any) -> T:
        return self.__class__(operator.le, '{} <= {}', [self, other])

    def __eq__(self: T, other: Any) -> T:
        return self.__class__(operator.eq, '{} == {}', [self, other])

    def __ne__(self: T, other: Any) -> T:
        return self.__class__(operator.ne, '{} != {}', [self, other])

    def __ge__(self: T, other: Any) -> T:
        return self.__class__(operator.ge, '{} >= {}', [self, other])

    def __gt__(self: T, other: Any) -> T:
        return self.__class__(operator.gt, '{} > {}', [self, other])
    # endregion

    # __pow__, __divmod__, __abs__ are special ones.
    # Not easy to cover those cases with codegen.
    def __pow__(self: T, other: Any, modulo: Any | None = None) -> T:
        if modulo is not None:
            return self.__class__(pow, 'pow({}, {}, {})', [self, other, modulo])
        return self.__class__(lambda a, b: a ** b, '{} ** {}', [self, other])

    def __rpow__(self: T, other: Any, modulo: Any | None = None) -> T:
        if modulo is not None:
            return self.__class__(pow, 'pow({}, {}, {})', [other, self, modulo])
        return self.__class__(lambda a, b: a ** b, '{} ** {}', [other, self])

    def __divmod__(self: T, other: Any) -> T:
        return self.__class__(divmod, 'divmod({}, {})', [self, other])

    def __rdivmod__(self: T, other: Any) -> T:
        return self.__class__(divmod, 'divmod({}, {})', [other, self])

    def __abs__(self: T) -> T:
        return self.__class__(abs, 'abs({})', [self])


class Symbol(SymbolicExpression):
    """
    Overwrite commonly-used arithmetic operators in Python, so that we can capture all the computations and replay them.
    """

    def expression_class(self):
        return SymbolicExpression

    def __init__(self, function: Callable[..., _cand] = cast(Callable[..., _cand], None),
                 repr_template: str = cast(str, None),
                 arguments: List[Any] = cast('List[MaybeChoice[_cand]]', None),
                 dry_run: bool = True):
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

    def forward(self) -> None:
        raise RuntimeError('You should never call forward of the composition of a value-choice.')

    def inner_choices(self) -> Iterable['ValueChoice']:
        """
        Return a generator of all leaf value choices.
        Useful for composition of value choices.
        No deduplication on labels. Mutators should take care.
        """
        for arg in self.arguments:
            if isinstance(arg, ValueChoiceX):
                yield from arg.inner_choices()

    def dry_run(self) -> _cand:
        """
        Dry run the value choice to get one of its possible evaluation results.
        """
        # values are not used
        return self._evaluate(iter([]), True)

    def all_options(self) -> Iterable[_cand]:
        """Explore all possibilities of a value choice.
        """
        # Record all inner choices: label -> candidates, no duplicates.
        dedup_inner_choices: Dict[str, List[_cand]] = {}
        # All labels of leaf nodes on tree, possibly duplicates.
        all_labels: List[str] = []

        for choice in self.inner_choices():
            all_labels.append(choice.label)
            if choice.label in dedup_inner_choices:
                if choice.candidates != dedup_inner_choices[choice.label]:
                    # check for choice with the same label
                    raise ValueError(f'"{choice.candidates}" is not equal to "{dedup_inner_choices[choice.label]}", '
                                     f'but they share the same label: {choice.label}')
            else:
                dedup_inner_choices[choice.label] = choice.candidates

        dedup_labels, dedup_candidates = list(dedup_inner_choices.keys()), list(dedup_inner_choices.values())

        for chosen in itertools.product(*dedup_candidates):
            chosen = dict(zip(dedup_labels, chosen))
            yield self.evaluate([chosen[label] for label in all_labels])

    def evaluate(self, values: Iterable[_cand]) -> _cand:
        """
        Evaluate the result of this group.
        ``values`` should in the same order of ``inner_choices()``.
        """
        return self._evaluate(iter(values), False)

    def _evaluate(self, values: Iterator[_cand], dry_run: bool = False) -> _cand:
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

    def __repr__(self) -> str:
        reprs = []
        for arg in self.arguments:
            if isinstance(arg, ValueChoiceX) and not isinstance(arg, ValueChoice):
                reprs.append('(' + repr(arg) + ')')  # add parenthesis for operator priority
            else:
                reprs.append(repr(arg))
        return self.repr_template.format(*reprs)
