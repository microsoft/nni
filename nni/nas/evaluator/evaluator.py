# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['Evaluator']

import abc
from typing import Any, Callable, Type, Union, cast


class Evaluator(abc.ABC):
    """
    Evaluator of a model. An evaluator should define where the training code is, and the configuration of
    training code. The configuration includes basic runtime information trainer needs to know (such as number of GPUs)
    or tune-able parameters (such as learning rate), depending on the implementation of training code.

    Each config should define how it is interpreted in ``_execute()``, taking only one argument which is the mutated model class.
    For example, functional evaluator might directly import the function and call the function.
    """

    def evaluate(self, model_cls: Union[Callable[[], Any], Any]) -> Any:
        """To run evaluation of a model. The model could be either a concrete model or a callable returning a model.

        The concrete implementation of evaluate depends on the implementation of ``_execute()`` in sub-class.
        """
        return self._execute(model_cls)

    def __repr__(self):
        items = ', '.join(['%s=%r' % (k, v) for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({items})'

    @staticmethod
    def _load(ir: Any) -> 'Evaluator':
        evaluator_type = ir.get('type')
        if isinstance(evaluator_type, str):
            # for debug purposes only
            for subclass in Evaluator.__subclasses__():
                if subclass.__name__ == evaluator_type:
                    evaluator_type = subclass
                    break
        assert issubclass(cast(type, evaluator_type), Evaluator)
        return cast(Type[Evaluator], evaluator_type)._load(ir)

    @abc.abstractmethod
    def _dump(self) -> Any:
        """
        Subclass implements ``_dump`` for their own serialization.
        They should return a dict, with a key ``type`` which equals ``self.__class__``,
        and optionally other keys.
        """
        pass

    @abc.abstractmethod
    def _execute(self, model_cls: Union[Callable[[], Any], Any]) -> Any:
        pass

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass
