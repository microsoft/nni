# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import traceback
from typing import Any

from nni.mutable import MutableExpression, Categorical, Numerical

_logger = logging.getLogger(__name__)


def conclude_assumptions(values: list[int | float]) -> dict[str, bool]:
    """Conclude some sympy assumptions based on the examples in values.

    Support assumptions are: positive, negative, nonpositive, nonnegative,
    zero, nonzero, odd, even, real.
    """
    if not values:
        return {}

    assumptions = {}
    assumptions['real'] = all(isinstance(v, (float, int)) for v in values)
    if not assumptions['real']:
        return assumptions

    assumptions['integer'] = all(isinstance(v, int) for v in values)
    if all(v > 0 for v in values):
        assumptions['positive'] = True
    if all(v < 0 for v in values):
        assumptions['negative'] = True
    if all(v >= 0 for v in values):
        assumptions['nonnegative'] = True
    if all(v <= 0 for v in values):
        assumptions['nonpositive'] = True
    if all(v == 0 for v in values):
        assumptions['zero'] = True
    if all(v != 0 for v in values):
        assumptions['nonzero'] = True

    if not assumptions['integer']:
        return assumptions

    if all(v % 2 == 0 for v in values):
        assumptions['even'] = True
    if all(v % 2 == 1 for v in values):
        assumptions['odd'] = True

    return assumptions


_seen_errors = set()


def expression_simplification(expression: MutableExpression):
    try:
        from sympy import Symbol, Expr, lambdify, simplify
    except ImportError:
        _logger.warning('sympy is not installed, give up expression simplification.')
        return expression

    # Get all symbols.
    mutables = expression.simplify()

    # Mutables will be substituted with sympy symbol / expressions.
    mutable_substitutes: dict[str, Symbol | Expr] = {}
    # (inverse substitution, in lambdify) Each sympy symbol corresponds to a mutable expression.
    inverse_substitutes: dict[Symbol | Expr, MutableExpression] = {}

    for name, mutable in mutables.items():
        if isinstance(mutable, Categorical):
            assumptions = conclude_assumptions(mutable.values)
            if not assumptions.get('real', False):
                _logger.warning('Expression simplification only supports categorical mutables with numerical choices, '
                                'but got %r. Give up.', mutable)

            # Variable substitution as sympy appears not very clever when handling odd/even.
            # Workaround: https://stackoverflow.com/q/75236716/6837658
            # It will leave some *2, //2 in the expression, but it's fine.
            odd = assumptions.pop('odd', False)

            if odd:
                if not assumptions.get('positive', False):
                    # Not all positive. This should be a rare case.
                    # That means we have -1 -> -3, -2 -> -5, ..., and we've lost 0 -> -1.

                    # There is no rule saying whether -1 is allowed or not, but there are rules about 0.
                    # We would simply say that "zero can exist" regardless of whether -1 appears in the sample.
                    if assumptions.get('nonnegative', False):
                        assumptions['nonpositive'] = True
                    assumptions.pop('nonzero', None)

                symbol = Symbol(name, **assumptions)
                mutable_substitutes[name] = symbol * 2 - 1  # 1 -> 1, 2 -> 3, 3 -> 5, ...
                inverse_substitutes[symbol] = (mutable + 1) // 2

                # I can't say whether we should do the same to even.

            else:
                symbol = Symbol(name, **assumptions)
                mutable_substitutes[name] = symbol
                inverse_substitutes[symbol] = mutable

        elif isinstance(mutable, Numerical):
            symbol = Symbol(name, real=True)  # we only know it's real
            mutable_substitutes[name] = symbol
            inverse_substitutes[symbol] = mutable

        else:
            _logger.warning('Expression simplification only supports categorical and numerical mutables, '
                            'but got %s in expression. Give up.',
                            type(mutable))
            return expression

    try:
        # Create sympy expression and simplify it.
        sym_expression = expression.evaluate(mutable_substitutes)
        simplified_sym_expression = simplify(sym_expression)

        # Convert sympy expression back to MutableExpression.
        # Can't use .sub directly here because `__index__` seems unhappy with sympy.
        simplified_fn = lambdify(list(inverse_substitutes.keys()), simplified_sym_expression)
        simplified_expr = simplified_fn(*inverse_substitutes.values())

        # Convert the expression to expected type.
        expected_type = type(expression.default())
        actual_type = type(simplified_expr.default() if isinstance(simplified_expr, MutableExpression) else simplified_expr)
        if actual_type != expected_type:
            if expected_type == int:
                simplified_expr = round(simplified_expr)
            elif expected_type == float:
                simplified_expr = MutableExpression.to_float(simplified_expr)
            else:
                _logger.warning('Simplified expression is of type %s, but expected type is %s. Cannot convert.',
                                actual_type, expected_type)
                return expression

    except Exception as e:
        error_repr = repr(e)
        if error_repr not in _seen_errors:
            _seen_errors.add(error_repr)
            _logger.warning('Expression simplification failed: %s. Give up.\nExpression: %s\n%s',
                            error_repr, expression, traceback.format_exc())
        else:
            # Similar errors have shown already.
            pass
        return expression

    return simplified_expr


def recursive_simplification(obj: Any) -> Any:
    """Simplify all expressions in obj recursively."""
    from .shape import MutableShape

    if isinstance(obj, MutableExpression):
        return expression_simplification(obj)
    elif isinstance(obj, MutableShape):
        return MutableShape(*[recursive_simplification(v) for v in obj])
    elif isinstance(obj, dict):
        return {k: recursive_simplification(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_simplification(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_simplification(v) for v in obj)
    return obj
