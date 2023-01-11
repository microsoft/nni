# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

"""Space definitions that are friendly to NAS executions.

All model spaces should inherit :class:`BaseModelSpace`, which then divides into two categories.

1. :class:`ExecutableModelSpace`, which will be the (converted) model space that is used in NAS executions.
2. Model space coupled with deep learning framework (e.g., :class:`nni.nas.nn.pytorch.ModelSpace`).

Type 2 will be converted to type 1 upon the launch of a NAS experiment.
"""

from nni.mutable.exception import SampleValidationError

__all__ = ['ModelStatus', 'BaseModelSpace', 'ExecutableModelSpace', 'RawFormatModelSpace', 'SimplifiedModelSpace']

from copy import deepcopy
from enum import Enum
from typing import NoReturn, Any, Callable, Iterable

from nni.common.serializer import is_traceable, SerializableObject
from nni.nas.evaluator import Evaluator
from nni.mutable import Mutable, Sample, MutableDict, LabeledMutable, frozen_factory
from nni.typehint import TrialMetric
from .frozen import model_context

from .metrics import Metrics


class BaseModelSpace(Mutable):
    """A model space is a collection of mutables, organized in a meaningful way (i.e., in a model way).

    :class:`BaseModelSpace` is almost only used for isinstance check.
    A few utility functions might be provided inside this class for convenience.
    """

    @classmethod
    def frozen_factory(cls, sample: Sample) -> frozen_factory:
        """Get a factory that creates a frozen model from this model space."""
        return frozen_factory(cls, model_context(sample))


class ModelStatus(str, Enum):
    """
    The status of model space.

    A model space is created in `Initialized` status.
    When the model space starts to mutate and is becoming a single model, the status will be set to `Mutating`.
    As the model space will share the same class with the mutated single model,
    the status flag is a useful indication for the difference between the two.

    When the mutation is done and the model get ready to train, its status becomes `Frozen`.
    Only `Frozen` models can be submitted to execution engine for training.
    When training started, the model's status becomes `Training`.
    If training is successfully ended, model's `metric` attribute get set and its status becomes `Trained`.
    If training failed, the status becomes `Failed`.
    """

    Initialized = "initialized"
    Mutating = "mutating"
    Frozen = "frozen"
    Training = "training"
    Trained = "trained"
    Failed = "failed"
    Interrupted = "interrupted"

    def __repr__(self):
        return f'{self.__class__.__name__}.{self.name}'

    def frozen(self):
        """Frozen model cannot be mutated any more."""
        return self in [ModelStatus.Frozen, ModelStatus.Training, ModelStatus.Trained, ModelStatus.Interrupted, ModelStatus.Failed]

    def completed(self):
        """Completed model status won't change any more."""
        return self in [ModelStatus.Trained, ModelStatus.Failed, ModelStatus.Interrupted]


class ExecutableModelSpace(BaseModelSpace):
    """Model space with an extra execute method that defines how the models should be evaluated.
    It should be ``ModelSpaceWithExecution`` but that's too long.

    Both model space, as well as single models mutated from the space,
    will be instances of :class:`ExecutableModelSpace`.
    They only differ in the status flag (see :class:`ModelStatus`).

    Since the single models that are directly evaluated are also of this type,
    this class has an :meth:`execute` method which defines how the training pipeline works based on the data in the instance.
    The models sent to execution engine for training should be frozen models and also instances of this class.

    In most cases, :class:`ExecutableModelSpace` only contains the necessary information
    that is required for NAS mutations and reconstruction of the original model.
    This makes the model space light-weighted, and easy to be serialized for sending to clusters.
    It also reforms the space to be more friendly to NAS algorithms (e.g., in the format of graphs).
    """

    status: ModelStatus
    """The status of the model space / model."""

    metrics: Metrics
    """The evaluation metrics of the model."""

    evaluator: Evaluator | None
    """Evaluator that assesses the quality of the model."""

    sample: Sample | None
    """The sample that is used to freeze this model. It's useful for debug and visualization.
    It could be left unset if sample is not used when freezing the model.

    It's supposed to be a dict which is previously known as **architecture dict**
    (however it can sometimes contain information about evaluator as well).

    Subclasses should set this attribute in :meth:`freeze` if they want to use it.
    They may also set a sample different from what they received in :meth:`freeze` if it's intended.
    """

    def __init__(self, status: ModelStatus = ModelStatus.Initialized) -> None:
        self.status = status
        self.metrics = Metrics()

    def execute(self) -> Any:
        """Execute the training (and/or evaluation)."""
        if self.evaluator is None:
            raise ValueError('Evaluator is not set, but default execute requires an evaluator.')
        return self.evaluator._execute(self)

    @classmethod
    def from_model(cls, model_space: BaseModelSpace, evaluator: Evaluator | None = None, **configs: Any) -> ExecutableModelSpace:
        """Convert any model space to a specific type of executable model space.

        Parameters
        ----------
        model_space
            Model space written in deep learning framework in most cases.
        evaluator
            A model usually requires an evaluator to be *executable*.
            But evaluator can sometimes be optional for debug purposes or to support fancy algorithms.
        configs
            Additional configurations for the executable model space.

        Returns
        -------
        The converted model space.
        """
        raise NotImplementedError('`from_model` is not implemented for {}'.format(cls.__name__))

    def executable_model(self) -> Any:
        """Fully instantiate the deep learning model (e.g., PyTorch Module) so that it's ready to be executed.

        :meth:`executable_model` is usually symmetrical to :meth:`from_model`.
        While :meth:`from_model` converts deep learning model to :class:`ExecutableModelSpace`,
        :meth:`executable_model` converts :class:`ExecutableModelSpace` back to deep learning model.

        Returns
        -------
        Typical this method should return a PyTorch / Tensorflow model (or model factory),
        depending on the input format of evaluator.
        """
        raise NotImplementedError('`executable_model` is not implemented for {}'.format(self.__class__.__name__))

    @property
    def metric(self) -> TrialMetric | None:
        """Training result of the model, or ``None`` if it's not yet trained or has failed to train."""
        return self.metrics.final


class RawFormatModelSpace(ExecutableModelSpace):
    """Model space that keeps the original model and does no conversion of model format
    (in contrast to :class:`SimplifiedModelSpace` or :class:`~nni.nas.space.GraphModelSpace`).

    It's possible that strategies directly operate on this format of model space,
    but it will be very slow (since dealing with deep learning models directly) and inflexible.

    Therefore, this is almost only useful when strategies need to fuse the model space and evaluator,
    which requires source-code-level access to those two components.
    One typical use case is one-shot strategy.

    In the current version, :class:`RawFormatModelSpace` can't be serialized and sent to remote machines.

    TODO: examples?
    """

    def __init__(self, model_space: BaseModelSpace, evaluator: Evaluator) -> None:
        super().__init__()
        self.model_space = model_space
        self.evaluator = evaluator
        self.sample = None

    def extra_repr(self) -> str:
        return f'model_space={self.model_space!r}, ' + \
            f'evaluator={self.evaluator!r}, ' + \
            (f'sample={self.sample!r}, ' if self.sample else '') + \
            (f'metrics={self.metrics!r}, ' if self.metrics else '') + \
            f'status={self.status!r}'

    @classmethod
    def from_model(cls, model_space: BaseModelSpace, evaluator: Evaluator | None = None, **configs) -> ExecutableModelSpace:
        return cls(model_space, evaluator)

    def freeze(self, sample: Sample) -> RawFormatModelSpace:
        if self.status != ModelStatus.Initialized:
            raise RuntimeError('Cannot freeze a model space that is not initialized.')
        self.validate(sample)

        new_model = RawFormatModelSpace(
            self.model_space.freeze(sample),
            self.evaluator.freeze(sample) if isinstance(self.evaluator, Mutable) else self.evaluator
        )
        new_model.sample = sample
        new_model.status = ModelStatus.Frozen
        return new_model

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        exception = self.model_space.check_contains(sample)
        if exception is not None:
            exception.paths.append('model')
            return exception
        if isinstance(self.evaluator, Mutable):
            exception = self.evaluator.check_contains(sample)
            if exception is not None:
                exception.paths.append('evaluator')
                return exception
        return None

    def executable_model(self) -> Any:
        return self.model_space

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        yield from self.model_space.leaf_mutables(is_leaf)
        if isinstance(self.evaluator, Mutable):
            yield from self.evaluator.leaf_mutables(is_leaf)

    def _dump(self) -> NoReturn:
        """Serialization of a :class:`RawFormatModelSpace` is not supported.

        Notes
        -----
        The potential issues with serialization are in two folds:
        
        1. The model space could be a deep learning model, and have been arbitrarily mutated by the strategy (e.g., one-shot).
           For example, one submodule is replaced by another, or a layer is removed.
           In this case, we surely cannot use the init arguments to recover the model.
        2. The model space could contain parameters (weights), that are meant to be part of the model space.
           (That's why we have :class:`RawFormatModelSpace` other than :class:`SimplifiedModelSpace`).
           In this case, we need to dump all the parameters, which could be potentially large and slow.

        One potential solution to this problem might be introducing an advanced version of ``nni.trace``,
        that allows users / strategies to define a function that recreates the current instance from scratch.
        This function must be runnable in a completely new isolated process.

        Another potential solution might be introducing several flags to evaluator to specific needs like one-shot.
        But I don't think it's a good idea, because I want to make the evaluator semantically simple.
        """
        raise NotImplementedError('`_dump` is not implemented for {}'.format(self.__class__.__name__))

    @classmethod
    def _load(cls, **kwargs) -> NoReturn:
        raise NotImplementedError('`_load` is not implemented for {}'.format(RawFormatModelSpace.__name__))


class SimplifiedModelSpace(ExecutableModelSpace):
    """Model space that is simplified (see :meth:`~nni.mutable.Mutable.simplify`),
    and only keeps the key information.

    With :class:`SimplifiedModelSpace`, all details inside the model will be removed,
    which means, the weights, attributes, inplace modifications of the model will all be lost.
    Only the simplified mutables and necessary init arguments to recover the model for execution will be kept.

    To work with :class:`SimplifiedModelSpace`,
    the model itself should detect :meth:`~nni.nas.space.current_model` context,
    and init a sampled concrete model when it's inside the context.
    Since the model will be recreated, ``freeze`` and ``contains`` method of model space is never used.
    """

    def __init__(self, model: Any, mutables: dict[str, Any] | MutableDict, evaluator: Evaluator) -> None:
        super().__init__()
        assert is_traceable(model), 'Model must be traceable.'
        self.model = model.trace_copy()  # Make a trace copy for recovery.
        if isinstance(mutables, MutableDict):
            self.mutables = mutables
        else:
            self.mutables = MutableDict(mutables)
        self.evaluator = evaluator

        self.sample: Sample | None = None  # only available when status is not mutating or fixed

    @classmethod
    def from_model(cls, model_space: BaseModelSpace, evaluator: Evaluator | None = None, **configs) -> ExecutableModelSpace:
        return cls(model_space, model_space.simplify(), evaluator)

    def freeze(self, sample: Sample) -> SimplifiedModelSpace:
        if self.status != ModelStatus.Initialized:
            raise RuntimeError('Cannot freeze a model space that is not initialized.')
        self.validate(sample)
        # Copy the current instance
        model = self.__class__(self.model, self.mutables, self.evaluator)
        # Set status and sample
        model.status = ModelStatus.Frozen
        model.sample = deepcopy(sample)
        # If evaluator is a mutable, freeze it here.
        if isinstance(self.evaluator, Mutable):
            model.evaluator = self.evaluator.freeze(sample)
        return model

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        exception = self.mutables.check_contains(sample)
        if exception is not None:
            exception.paths.append('model')
            return exception
        if isinstance(self.evaluator, Mutable):
            exception = self.evaluator.check_contains(sample)
            if exception is not None:
                exception.paths.append('evaluator')
                return exception
        return None

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        yield from self.mutables.leaf_mutables(is_leaf)
        if isinstance(self.evaluator, Mutable):
            yield from self.evaluator.leaf_mutables(is_leaf)

    def extra_repr(self) -> str:
        return f'model={self.model}, mutables={self.mutables}, evaluator={self.evaluator}, ' + \
            (f'sample={self.sample!r}, ' if self.sample else '') + \
            (f'metrics={self.metrics!r}, ' if self.metrics else '') + \
            f'status={self.status!r}'

    def executable_model(self) -> Any:
        if self.sample is None:
            raise RuntimeError('Cannot get executable model from a model space that is not frozen.')
        with model_context(self.sample):
            # If it's in the same process, we need to re-initialize it. Therefore, `trace_copy()`.
            # If it's in another process, we only have a symbol and arguments. Therefore `get()`.
            # Note that `get()` might not be available for every traceable, but should work for `trace_copy()` results in this case.
            # We don't insist traceable to be true. Otherwise it will create another subclass of ModelSpace,
            # which ruins the label namespaces' numbering.
            return self.model.trace_copy().get(traceable=False)

    def _dump(self) -> dict:
        rv = {
            'status': self.status,
            # Have to break apart the model here.
            # Otherwise it will be instantiated immediately when loading, which is not what we want.
            'model_symbol': self.model.trace_symbol,
            'model_args': self.model.trace_args,
            'model_kwargs': self.model.trace_kwargs,
            'evaluator': self.evaluator,
        }
        if self.status != ModelStatus.Initialized:
            rv['sample'] = self.sample
            rv['metrics'] = self.metrics
        else:
            rv['mutables'] = self.mutables
        return rv

    @classmethod
    def _load(cls, **attrs) -> SimplifiedModelSpace:
        rv = SimplifiedModelSpace(
            SerializableObject(attrs['model_symbol'], attrs['model_args'], attrs['model_kwargs']),
            attrs['mutables'] if attrs['status'] == ModelStatus.Initialized else {},
            attrs['evaluator'],
        )
        rv.status = attrs['status']
        if 'sample' in attrs:
            rv.sample = attrs['sample']
        if 'metrics' in attrs:
            rv.metrics = attrs['metrics']
        return rv
