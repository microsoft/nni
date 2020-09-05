from ..strategy import Sampler, _FixedSampler


class Mutator:
    """
    Mutators should override `retrieve_targeted_graph`(?) and `mutate` methods.
    They can use `choice` to ask for decision from sampler.

    Strategy should use `apply` to mutate graphs.
    If the sampler needs candidates information before making choice,
    `dry_run` can be used to pre-fetch candidates.
    """
    def __init__(self) -> None:
        self._base_graph = None
        self._sampler = None

    def retrieve_targeted_graph(self, graph: 'Graph') -> 'Graph':
        raise NotImplementedError()

    def mutate(self, graph: 'Graph') -> None:
        raise NotImplementedError()

    def apply(self, graph: 'Graph', sampler_or_choices: 'Union[Sampler, Iterable[Any]]') -> 'Graph':
        self._base_graph = graph
        if isinstance(sampler_or_choices, Sampler):
            self._sampler = sampler_or_choices
        else:
            self._sampler = _FixedSampler(sampler_or_choices)
        graph = self._base_graph.duplicate()
        self.mutate(graph)
        return graph

    def choice(self, candidates: 'Iterable[Any]') -> 'Any':
        return self._sampler.choice(list(candidates))

    def dry_run(self, graph: 'Graph') -> 'List[List[Any]]':
        """
        Dry-run this mutator on bound graph. 

        There is no guarantee whether the candidates are constant or not.
        It depends on implementation of `mutate`.

        Return
        ======
        List[List[Any]]
            Choice candidates asked by mutator during mutation.
        """
        recorder = _RecorderSampler()
        new_graph = self.apply(graph, recorder)
        return new_graph, recorder.recorded_candidates


class _RecorderSampler(Sampler):
    def __init__(self) -> None:
        self.recorded_candidates: 'List[List[Any]]' = []

    def choice(self, candidates: 'List[Any]') -> 'Any':
        self.recorded_candidates.append(candidates)
        return candidates[0]
