from . import utils


class Sampler:
    def choice(self, candidates: 'List[Any]') -> 'Any':
        raise NotImplementedError()


class _FixedSampler(Sampler):
    def __init__(self, choices: 'List[List[Any]]') -> None:
        self._rest_choices = choices

    def choice(self, candidates: 'List[Any]') -> 'Any':
        return self._rest_choices.pop(0)


# TODO
class BaseStrategy:
    def __call__(self):
        training_graphs = []
        while True:
            trained_graphs = [graph for graph in training_graphs if graph.metrics is not None]
            training_graphs = [graph for graph in training_graphs if graph not in trained_graphs]

            for graph in trained_graphs:
                self.on_receive_feedbacks('final', graph.metrics)

            idle = utils.experiment_config()['concurrency'] - len(training_graphs)
            self.on_have_avail_resource(self, idle)


    def on_have_avail_resource(self, amount_resource):
        raise NotImplementedError

    def on_receive_feedbacks(self, type, results):
        pass
