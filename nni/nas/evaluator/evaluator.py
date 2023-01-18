# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['Evaluator', 'MutableEvaluator', 'FrozenEvaluator']

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Iterable, Generator, TYPE_CHECKING, cast
from typing_extensions import Literal

import nni
from nni.common.serializer import is_traceable, SerializableObject
from nni.mutable import Mutable, Sample, LabeledMutable, SampleValidationError
from nni.mutable.mutable import _mutable_equal
from nni.runtime.trial_command_channel import TrialCommandChannel, set_default_trial_command_channel, get_default_trial_command_channel
from nni.typehint import ParameterRecord, TrialMetric, Parameters

if TYPE_CHECKING:
    from nni.nas.space import ExecutableModelSpace

_logger = logging.getLogger(__name__)


class Evaluator:
    """Base class of evaluator.

    To users, the evaluator is to assess the quality of a model and return a score.
    When an evaluator is defined, it usually accepts a few arguments, such as
    basic runtime information (e.g., whether to use GPU), dataset used,
    as well as hyper-parameters (such as learning rate).
    These parameters can be sometimes tunable and searched by algorithms (see :class:`MutableEvaluator`).

    Different evaluators could have different use scenarios and requirements on the model.
    For example, :class:`~nni.nas.evaluator.pytorch.Classification` is tailored for classification models,
    and assumes the model has a ``forward`` method that takes a batch of data and returns logits.
    Evaluators might also have different assumptions, some of which are requirements of certain algorithms.
    The evaluator with the most freedom is :class:`~nni.nas.evaluator.FunctionalEvaluator`,
    but it's also incompatible with some algorithms.

    To developers, the evaluator is to implement *all the logics involving forward/backward of neural networks*.
    Sometimes the algorithm requires the training and searching at the same time (e.g., one-shot algos).
    In that case, although the searching part doesn't logically belong to the evaluator,
    it is still the evaluator's responsibility to implement it,
    and the search algorithms will make sure to properly manipulate the evaluator to achieve the goal.

    .. tip::

        Inside evaluator, you can use standard :doc:`NNI trial APIs </reference/hpo>` to communicate with the exploration strategy.
        Common usages include:

        1. Use :func:`nni.get_current_parameter` to get the current :class:`~nni.nas.space.ExecutableModelSpace`.
           Notice that :class:`~nni.nas.space.ExecutableModelSpace` is not a directly-runnable model (e.g., a PyTorch model),
           which is different from the model received in :meth:`evaluate`.
           :class:`~nni.nas.space.ExecutableModelSpace` objects are useful for debugging, as well as for some evaluators
           which need to know extra details about how the model is sampled.
        2. Use :func:`nni.report_intermediate_result` to report intermediate results.
        3. Use :func:`nni.report_final_result` to report final results.

        These APIs are only available when the evaluator is executed by NNI.
        We recommend using ``nni.get_current_parameter() is not None`` to check if the APIs are available before using them.
        Please AVOID using :func:`nni.get_next_parameter()` because NAS framework has already handled the logic
        of retrieving the next parameter. Incorrectly using :func:`nni.get_next_parameter()` may cause unexpected behavior.
    """

    def evaluate(self, model: Any) -> Any:
        """To run evaluation of a model. The model is usually a concrete model.
        The return value of :meth:`evaluate` can be anything. Typically it's used for test purposes.

        Subclass should override this.
        """
        raise NotImplementedError()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    @staticmethod
    @contextmanager
    def mock_runtime(model: ExecutableModelSpace) -> Generator[None, None, None]:
        """
        Context manager to mock trial APIs for standalone usage.

        Under the with-context of this method, :func:`nni.get_current_parameter` will return the given model.

        NOTE: This method might become a utility in trial command channel in future.

        Parameters
        ----------
        model
            The model to be evaluated. It should be a :class:`~nni.nas.space.ExecutableModelSpace` object.

        Examples
        --------
        This method should be mostly used when testing a evaluator.
        A typical use case is as follows:

        >>> frozen_model = model_space.freeze(sample)
        >>> with evaluator.mock_runtime(frozen_model):
        ...     evaluator.evaluate(frozen_model.executable_model())
        """

        if nni.get_current_parameter() is not None:
            raise RuntimeError("Cannot mock trial when trial APIs are already available.")

        from nni.nas.space import ExecutableModelSpace
        if not isinstance(model, ExecutableModelSpace):
            raise TypeError("model should be an ExecutableModelSpace object.")

        trial_command_channel = get_default_trial_command_channel()
        original_params = nni.trial._params
        original_seq = nni.trial._intermediate_seq

        try:
            set_default_trial_command_channel(_EvaluatorMockTrialCommandChannel(model))
            assert nni.get_next_parameter() is model
            yield
        finally:
            set_default_trial_command_channel(trial_command_channel)
            # NOTE: It might have some side effects on nni.trial._params.
            #       Might cause the trial_available() to still return true after the context.
            #       The following hack might address the problem.
            nni.trial._params = original_params
            nni.trial._intermediate_seq = original_seq

    @staticmethod
    def _load(**ir: Any) -> 'Evaluator':
        """Subclass implements ``_load`` for their own serialization."""
        raise NotImplementedError()

    def _dump(self) -> Any:
        """Subclass implements ``_dump`` for their own serialization."""
        raise NotImplementedError()

    # def _execute(self, model: ExecutableModelSpace) -> Any:
    def _execute(self, model: Any) -> Any:
        """Advanced users can overwrite this to avoid instantiation of the deep learning model.

        For internal uses only.
        """
        executable_model = model.executable_model()
        return self.evaluate(executable_model)


class MutableEvaluator(Mutable, Evaluator):
    """
    Evaluators with tunable parameters by itself (e.g., learning rate).

    The tunable parameters must be an argument of the evaluator's instantiation,
    or an argument of the arguments' instantiation and etc.

    To use this feature, there are two requirements:

    1. The evaluator must inherit :class:`MutableEvaluator` rather than :class:`Evaluator`.
    2. Make sure the init arguments have been saved in ``trace_kwargs``,
       and the instance can be cloned with ``trace_copy``.
       The easiest way is to wrap the evaluator with :func:`nni.trace`.
       If the mutable parameter exists somewhere in the nested instantiation.
       All the levels must all be wrapped with :func:`nni.trace`.

    Examples
    --------
    >>> def get_data(shuffle): ...
    ...
    >>> @nni.trace                                  # 1. must wrap here
    ... class MyOwnEvaluator(MutableEvaluator):     # 2. must inherit MutableEvaluator
    ...     def __init__(self, lr, data): ...
    ...
    >>> evaluator = MyOwnEvaluator(
    ...     lr=Categorical([0.1, 0.01]),      # the argument can be tunable
    ...     data=nni.trace(get_data)(         # if there is mutable parameters inside, this must also have nni.trace
    ...         shuffle=Categorical([False, True])
    ...     )
    ... )
    >>> evaluator.simplify()
    {'global/1': Categorical([0.1, 0.01], label='global/1'), 'global/2': Categorical([False, True], label='global/2')}
    """

    @staticmethod
    def freeze_traceable_object(obj: Any, sample: dict[str, Any]) -> Any:
        # Could return:
        # 1. The same type of obj, if it is not traceable or nothing is changed.
        # 2. SerializableObject. If something is mutated, and the return value is merely an empty shell of parameters.
        if not is_traceable(obj, must_be_instance=True):
            return obj

        updates = {}

        # Iterate over all the arguments that have been used to instantiate the object
        for key, param in obj.trace_kwargs.items():
            if isinstance(param, Mutable):
                updates[key] = param.freeze(sample)
            elif is_traceable(param):
                # Recursively
                sub_update = MutableEvaluator.freeze_traceable_object(param, sample)
                if sub_update is not param:  # if mutated
                    updates[key] = sub_update

        if updates:
            mutated_obj = obj.trace_copy()                  # Make a copy
            mutated_obj.trace_kwargs.update(updates)        # Mutate

            # Should instantiate the full mutated object here.
            # But it's postponed to the next call of evaluate to save memory.
            # mutated_obj = mutated_obj.get()

            return mutated_obj

        return obj

    @staticmethod
    def expand_trace_kwargs(obj: Any) -> Iterator[Mutable]:
        if not is_traceable(obj, must_be_instance=True):
            raise TypeError(f'{obj} is not traceable.')
        for pos, param in enumerate(obj.trace_args):
            if isinstance(param, Mutable):
                raise ValueError(f'We currently do not support mutable parameters in positional arguments: {obj} (position: {pos})')
        for param in obj.trace_kwargs.values():
            if isinstance(param, Mutable):
                yield param
            # Recursively yield all the nested parameters
            if is_traceable(param, must_be_instance=True):
                yield from MutableEvaluator.expand_trace_kwargs(param)

    def freeze(self, sample: Sample) -> FrozenEvaluator | MutableEvaluator:
        """Upon freeze, :class:`MutableEvaluator` will freeze all the mutable parameters
        (as well as nested parameters),
        and return a :class:`FrozenEvaluator`.

        The evaluator will not be fully initialized to save the memory,
        especially when parameters contain large objects such as datasets.
        To use the evaluator, call :meth:`FrozenEvaluator.get` to get the full usable evaluator.

        Returns
        -------
        The frozen evaluator.
        """
        frozen_evaluator = self.freeze_traceable_object(self, sample)
        assert isinstance(frozen_evaluator, (MutableEvaluator, SerializableObject)), \
            'The evaluator must still be a MutableEvaluator or SerializableObject after freezing.'
        if type(frozen_evaluator) is SerializableObject:
            return FrozenEvaluator(frozen_evaluator)
        else:
            assert isinstance(frozen_evaluator, MutableEvaluator)
            return frozen_evaluator

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        for param in self.expand_trace_kwargs(self):
            yield from param.leaf_mutables(is_leaf)

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        for mutable in self.expand_trace_kwargs(self):
            exception = mutable.check_contains(sample)
            if exception is not None:
                exception.paths.append(str(mutable))
                return exception
        return None

    def is_mutable(self) -> bool:
        """Whether some arguments of the evaluator are mutable.

        Although the evaluator is mutable, it may contain no mutable parameters,
        i.e., all its parameters (including nested ones) are fixed values.
        Return false if there is none.
        """
        # It's a clever way to check whether an iterator is not empty, without expanding it.
        for _ in self.expand_trace_kwargs(self):
            return True
        return False

    @classmethod
    def _load(cls, **ir: Any) -> MutableEvaluator:
        return cls(**ir)  # type: ignore

    def _dump(self) -> dict:
        if not is_traceable(self):
            raise TypeError(f'{self} is not traceable.')
        return self.trace_kwargs  # type: ignore

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        if not is_traceable(self):
            raise TypeError(f'{self} is not traceable.')
        return _mutable_equal(self.trace_kwargs, other.trace_kwargs)  # type: ignore


class FrozenEvaluator(Evaluator):
    """:meth:`MutableEvaluator.freeze` returns a :class:`FrozenEvaluator`,
    which is purely an empty-shell (i.e., symbol and init parameters), and not instantiated.

    For the evaluator itself and its parameters, if its constructor is decorated with ``nni.trace``,
    in :attr:`frozen_obj`, it will be a :class:`SerializableObject` that contains the constructor (class / function)
    as well as the arguments that have been used to instantiate the object.

    When :meth:`evaluate` is invoked, it instantiates to the full evaluator recursively and execute it.
    """

    def __init__(self, frozen_obj: SerializableObject):
        self.frozen_obj = frozen_obj
        self._instance: Evaluator | None = None

    @staticmethod
    def recursive_instantiate(obj: Any) -> Any:
        if type(obj) is not SerializableObject:
            # It has been instantiated
            return obj

        # obj must be a SerializableObject here.
        updates = {}
        for key, param in obj.trace_kwargs.items():
            sub_update = FrozenEvaluator.recursive_instantiate(param)
            if sub_update is not param:
                updates[key] = sub_update

        if updates:
            obj.trace_kwargs.update(updates)

        # Calling get() of a SerializableObject gives a fully instantiated object.
        # We can't bypass this get() even if updates is empty because obj is never instantiated.
        return obj.get()

    def get(self) -> Evaluator:
        """Instantiate the full evaluator.

        The instantiated evaluator will be cached and reused next time.
        """
        if self._instance is not None:
            return self._instance
        self._instance = self.recursive_instantiate(self.frozen_obj)
        assert isinstance(self._instance, Evaluator), f'Instantiated evaluator must be an Evaluator, got {type(self._instance)}'
        return self._instance

    def evaluate(self, model: Any) -> Any:
        """Calling :meth:`get` and execute the evaluator."""
        return self.get().evaluate(model)

    def extra_repr(self) -> str:
        return f'frozen_obj={self.frozen_obj}, initialized={self._instance is not None}'

    @classmethod
    def _load(cls, **ir: Any) -> 'FrozenEvaluator':
        return ir['trace_symbol'](*ir['trace_args'], **ir['trace_kwargs'])

    def _dump(self) -> Any:
        return {
            'trace_symbol': self.trace_symbol,
            'trace_args': self.trace_args,
            'trace_kwargs': self.trace_kwargs,
        }

    # The following APIs are provided as shortcut of frozen_obj.

    @property
    def trace_symbol(self):
        return self.frozen_obj.trace_symbol

    @property
    def trace_args(self):
        return self.frozen_obj.trace_args

    @property
    def trace_kwargs(self):
        return self.frozen_obj.trace_kwargs

    def trace_copy(self):
        return FrozenEvaluator(self.frozen_obj.trace_copy())


class _EvaluatorMockTrialCommandChannel(TrialCommandChannel):
    """Mock a trial command channel for evaluator debugging."""

    def __init__(self, model: ExecutableModelSpace):
        self.model = model

    def receive_parameter(self) -> ParameterRecord | None:
        return {
            'parameter_id': 0,
            'parameters': cast(Parameters, self.model)
        }

    def send_metric(self, type: Literal['PERIODICAL', 'FINAL'], parameter_id: int | None,  # pylint: disable=redefined-builtin
                    trial_job_id: str, sequence: int, value: TrialMetric) -> None:
        if type == 'FINAL':
            self.model.metrics.final = value
            _logger.info('[Mock] Final metric: %s', value)
        else:
            self.model.metrics.add_intermediate(value)
            _logger.info('[Mock] Intermediate metric: %s', value)
