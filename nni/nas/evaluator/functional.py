# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from nni.common.serializer import SerializableObject
from .evaluator import MutableEvaluator


class FunctionalEvaluator(MutableEvaluator):
    """
    Functional evaluator that directly takes a function and thus should be general.
    See :class:`~nni.nas.evaluator.Evaluator` for instructions on how to write this function.

    Attributes
    ----------
    function
        The full name of the function.
    arguments
        Keyword arguments for the function other than model.
    """

    def __init__(self, function, **kwargs):
        self.function = function
        self.arguments = kwargs

    def extra_repr(self):
        return f"{self.function!r}, arguments={self.arguments!r})"

    # NOTE: FunctionalEvaluator implements the traceable interface by itself,
    #       so that it doesn't need the `nni.trace` decorator.
    #       But I guess it works with the decorator as well.

    @property
    def trace_symbol(self):
        return self.__class__

    @property
    def trace_args(self):
        return []

    @property
    def trace_kwargs(self):
        return {
            'function': self.function,
            **self.arguments
        }

    def trace_copy(self):
        return SerializableObject(self.__class__, [], self.trace_kwargs)

    def evaluate(self, model):
        return self.function(model, **self.arguments)
