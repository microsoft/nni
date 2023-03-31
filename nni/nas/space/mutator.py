# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterable, List, Optional, TYPE_CHECKING, Iterator

from numpy.random import RandomState

from nni.mutable import (
    LabeledMutable, MutableList, MutableDict, Categorical, Mutable, SampleValidationError,
    Sample, SampleMissingError, label_scope, auto_label, frozen_context
)

from .space import ModelStatus

if TYPE_CHECKING:
    from .graph import GraphModelSpace

__all__ = ['MutationSampler', 'Mutator', 'StationaryMutator', 'InvalidMutation', 'MutatorSequence', 'Mutation']


Choice = Any


class MutationSampler:
    """
    Handles :meth:`Mutator.choice` calls.

    Choice is the only supported type for mutator.
    """

    def choice(self, candidates: List[Choice], mutator: 'Mutator', model: GraphModelSpace, index: int) -> Choice:
        raise NotImplementedError()

    def mutation_start(self, mutator: 'Mutator', model: GraphModelSpace) -> None:
        pass

    def mutation_end(self, mutator: 'Mutator', model: GraphModelSpace) -> None:
        pass


class Mutator(LabeledMutable):
    """
    Mutates graphs in model to generate new model.

    By default, mutator simplifies to a single-value dict with its own label as key, and itself as value.
    At freeze, the strategy should provide a :class:`MutationSampler` in the dict.
    This is because the freezing of mutator is dynamic
    (i.e., requires a variational number of random numbers, dynamic ranges for each random number),
    and the :class:`MutationSampler` here can be considered as some random number generator
    to produce a random sequence based on the asks in :meth:`Mutator.mutate`.

    On the other hand, a subclass mutator should implement :meth:`Mutator.mutate`, which calls :meth:`Mutator.choice` inside,
    and :meth:`Mutator.choice` invokes the bounded sampler to "random" a choice.

    The label of the mutator in most cases is the label of the nodes on which the mutator is applied to.

    I imagine that mutating any model space (other than graph) might be useful,
    but we would postpone the support to when we actually need it.
    """

    def __init__(self, *, sampler: Optional[MutationSampler] = None, label: Optional[str] = None):
        self.sampler: Optional[MutationSampler] = sampler
        self.label: str = auto_label(label)
        self.model: Optional[GraphModelSpace] = None
        self._cur_model: Optional[GraphModelSpace] = None
        self._cur_choice_idx: Optional[int] = None

    def extra_repr(self) -> str:
        return f'label={self.label!r}'

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        """By default, treat self as a whole labeled mutable in the format dict.

        Sub-class can override this to dry run the mutation upon the model and return the mutated model
        for the followed-up dry run.

        See Also
        --------
        nni.mutable.Mutable.leaf_mutables
        """
        # Same as `leaf_mutables` in LabeledMutable.
        return super().leaf_mutables(is_leaf)

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        """Check if the sample is valid for this mutator.

        See Also
        --------
        nni.mutable.Mutable.check_contains
        """
        if self.label not in sample:
            return SampleMissingError(f"Mutator {self.label} not found in sample.")
        if not isinstance(sample[self.label], MutationSampler):
            return SampleValidationError(f"Mutator {self.label} is not a MutationSampler.")
        return None

    def freeze(self, sample: dict[str, Any]) -> GraphModelSpace:
        """When freezing a mutator, we need a model to mutate on, as well as a sampler to generate choices.

        As how many times the mutator is applied on the model is often variational,
        a sample with fixed length will not work.
        The dict values in ``sample`` should be a sampler inheriting :class:`MutationSampler`.
        But there are also cases where ``simplify()`` converts the mutation process into some fixed operations
        (e.g., in :class:`StationaryMutator`).
        In this case, sub-class should handle the freeze logic on their own.

        :meth:`Mutator.freeze` needs to be called in a ``bind_model`` context.
        """
        self.validate(sample)
        assert self.model is not None, 'Mutator must be bound to a model before freezing.'
        return self.bind_sampler(sample[self.label]).apply(self.model)

    def bind_sampler(self, sampler: MutationSampler) -> Mutator:
        """Set the sampler which will handle :meth:`Mutator.choice` calls."""
        self.sampler = sampler
        return self

    @contextmanager
    def bind_model(self, model: GraphModelSpace) -> Iterator[Mutator]:
        """Mutators need a model, based on which they generate new models.
        This context manager binds a model to the mutator, and unbinds it after the context.

        Examples
        --------
        >>> with mutator.bind_model(model):
        ...     mutator.simplify()
        """
        try:
            self.model = model
            yield self
        finally:
            self.model = None

    def apply(self, model: GraphModelSpace) -> GraphModelSpace:
        """
        Apply this mutator on a model.
        The model will be copied before mutation and the original model will not be modified.

        Returns
        -------
        The mutated model.
        """
        assert self.sampler is not None
        copy = model.fork()
        copy.status = ModelStatus.Mutating
        self._cur_model = copy
        self._cur_choice_idx = 0
        self._cur_samples = []

        # Some mutate() requires a full mutation history of the model.
        # Therefore, parent needs to be set before the mutation.
        copy.parent = Mutation(self, self._cur_samples, model, copy)
        self.sampler.mutation_start(self, copy)
        self.mutate(copy)
        self.sampler.mutation_end(self, copy)
        self._cur_model = None
        self._cur_choice_idx = None
        return copy

    def mutate(self, model: GraphModelSpace) -> None:
        """
        Abstract method to be implemented by subclass.
        Mutate a model in place.
        """
        raise NotImplementedError()

    def choice(self, candidates: Iterable[Choice]) -> Choice:
        """Ask sampler to make a choice."""
        assert self.sampler is not None and self._cur_model is not None and self._cur_choice_idx is not None
        ret = self.sampler.choice(list(candidates), self, self._cur_model, self._cur_choice_idx)
        self._cur_samples.append(ret)
        self._cur_choice_idx += 1
        return ret

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> GraphModelSpace | None:
        """Use a :class:`_RandomSampler` that generates a random sample when mutates.

        See Also
        --------
        nni.mutable.Mutable.random
        """
        sample: Sample = {} if memo is None else memo
        if random_state is None:
            random_state = RandomState()
        if self.label not in sample:
            sample[self.label] = _RandomSampler(random_state)
        if self.model is not None:
            # Model is binded, perform the freeze.
            return self.freeze(sample)
        else:
            # This will only affect the memo.
            # Parent random will take care of the freeze afterwards.
            return None


class StationaryMutator(Mutator):
    """A mutator that can be dry run.

    :class:`StationaryMutator` invoke :class:`StationaryMutator.dry_run` to predict choice candidates,
    such that the mutator simplifies to some static choices within `simplify()`.
    This could be convenient to certain algorithms which do not want to handle dynamic samplers.
    """

    def __init__(self, *, sampler: Optional[MutationSampler] = None, label: Optional[str] = None):
        super().__init__(sampler=sampler, label=label)
        self._dry_run_choices: Optional[MutableDict] = None

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        """Simplify this mutator to a number of static choices. Invokes :meth:`StationaryMutator.dry_run`.

        Must be wrapped in a ``bind_model`` context.
        """
        assert self.model is not None, 'Mutator must be bound to a model before calling `simplify()`.'
        choices, model = self.dry_run(self.model)
        self._dry_run_choices = MutableDict(choices)
        yield from self._dry_run_choices.leaf_mutables(is_leaf)
        self.model = model

    def check_contains(self, sample: dict[str, Any]):
        if self._dry_run_choices is None:
            raise RuntimeError(
                'Dry run choices not found. '
                'Graph model space with stationary mutators must first invoke `simplify()` before freezing.'
            )
        return self._dry_run_choices.check_contains(sample)

    def freeze(self, sample: dict[str, Any]) -> GraphModelSpace:
        self.validate(sample)

        assert self._dry_run_choices is not None
        assert self.model is not None

        # The orders should be preserved here
        samples = [sample[label] for label in self._dry_run_choices]
        # We fake a FixedSampler in this freeze to consume the already-generated samples.s
        sampler = _FixedSampler(samples)
        return self.bind_sampler(sampler).apply(self.model)

    def dry_run(self, model: GraphModelSpace) -> tuple[dict[str, Categorical], GraphModelSpace]:
        """Dry run mutator on a model to collect choice candidates.

        If you invoke this method multiple times on same or different models,
        it may or may not return identical results, depending on how the subclass implements `Mutator.mutate()`.

        Recommended to be used in :meth:`simplify` if the mutator is static.
        """
        sampler_backup = self.sampler
        recorder = _RecorderSampler()
        self.sampler = recorder
        new_model = self.apply(model)
        self.sampler = sampler_backup

        # Local import to avoid name conflict.
        from nni.mutable.utils import label
        # NOTE: This is hacky. It fakes a label object by splitting the label string.
        _label = label(self.label.split('/'))

        if len(recorder.recorded_candidates) != 1:
            # If the mutator is applied multiple times on the model (e.g., applied to multiple nodes)
            # choices can created with a suffix to distinguish them.

            with label_scope(_label):
                choices = [Categorical(candidates, label=str(i)) for i, candidates in enumerate(recorder.recorded_candidates)]
        else:
            # Only one choice.
            choices = [Categorical(recorder.recorded_candidates[0], label=_label)]
        return {c.label: c for c in choices}, new_model

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> GraphModelSpace | None:
        """Use :meth:`nni.mutable.Mutable.random` to generate a random sample."""
        return Mutable.random(self, memo, random_state)


class MutatorSequence(MutableList):
    """Apply a series of mutators on our model, sequentially.

    This could be generalized to a DAG indicating the dependencies between mutators,
    but we don't have a use case for that yet.
    """

    mutables: list[Mutator]

    def __init__(self, mutators: list[Mutator]):
        assert all(isinstance(mutator, Mutator) for mutator in mutators), 'mutators must be a list of Mutator'
        super().__init__(mutators)
        self.model: Optional[GraphModelSpace] = None

    @contextmanager
    def bind_model(self, model: GraphModelSpace) -> Iterator[MutatorSequence]:
        """Bind the model to a list of mutators.
        The model (as well as its successors) will be bounded to the mutators one by one.
        The model will be unbinded after the context.

        Examples
        --------
        >>> with mutator_list.bind_model(model):
        ...     mutator_list.freeze(samplers)
        """
        try:
            self.model = model
            yield self
        finally:
            self.model = None

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        assert self.model is not None, 'Mutator must be bound to a model before calling `simplify()`.'
        model = self.model
        with frozen_context():  # ensure_frozen() might be called inside
            for mutator in self.mutables:
                with mutator.bind_model(model):
                    yield from mutator.leaf_mutables(is_leaf)
                    model = mutator.model
                    assert model is not None

    def freeze(self, sample: dict[str, Any]) -> GraphModelSpace:
        assert self.model is not None, 'Mutator must be bound to a model before freezing.'
        model = self.model
        for mutator in self.mutables:
            with mutator.bind_model(model):
                model = mutator.freeze(sample)
        return model


class _RecorderSampler(MutationSampler):
    def __init__(self):
        self.recorded_candidates: List[List[Choice]] = []

    def choice(self, candidates: List[Choice], *args) -> Choice:
        self.recorded_candidates.append(candidates)
        return candidates[0]


class _FixedSampler(MutationSampler):
    def __init__(self, samples):
        self.samples = samples

    def choice(self, candidates, mutator, model, index):
        if not 0 <= index < len(self.samples):
            raise RuntimeError(f'Invalid index {index} for samples {self.samples}')
        if self.samples[index] not in candidates:
            raise RuntimeError(f'Invalid sample {self.samples[index]} for candidates {candidates}')
        return self.samples[index]


class _RandomSampler(MutationSampler):
    def __init__(self, random_state: RandomState):
        self.random_state = random_state

    def choice(self, candidates, mutator, model, index):
        return self.random_state.choice(candidates)


class InvalidMutation(SampleValidationError):
    pass


class Mutation:
    """
    An execution of mutation, which consists of four parts: a mutator, a list of decisions (choices),
    the model that it comes from, and the model that it becomes.

    In general cases, the mutation logs are not reliable and should not be replayed as the mutators can
    be arbitrarily complex. However, for inline mutations, the labels correspond to mutator labels here,
    this can be useful for metadata visualization and python execution mode.

    Attributes
    ----------
    mutator
        Mutator.
    samples
        Decisions/choices.
    from_
        Model that is comes from.
    to
        Model that it becomes.
    """

    def __init__(self, mutator: 'Mutator', samples: List[Any], from_: GraphModelSpace, to: GraphModelSpace):  # noqa: F821
        self.mutator: 'Mutator' = mutator  # noqa: F821
        self.samples: List[Any] = samples
        self.from_: GraphModelSpace = from_
        self.to: GraphModelSpace = to

    def __repr__(self):
        return f'Mutation(mutator={self.mutator}, samples={self.samples}, from={self.from_}, to={self.to})'
