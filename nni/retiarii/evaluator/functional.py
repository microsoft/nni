# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..graph import Evaluator


class FunctionalEvaluator(Evaluator):
    """
    Functional evaluator that directly takes a function and thus should be general.

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

    @staticmethod
    def _load(ir):
        return FunctionalEvaluator(ir['function'], **ir['arguments'])

    def _dump(self):
        return {
            'function': self.function,
            'arguments': self.arguments
        }

    def _execute(self, model_cls):
        return self.function(model_cls, **self.arguments)

    def __eq__(self, other):
        return self.function == other.function and self.arguments == other.arguments
