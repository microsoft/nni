import random
import logging
import sdk

_logger = logging.getLogger(__name__)

def main():
    """
    this one can be seen as a scheduler in strategy, and there is another component which decides the choices.
    for simplicity, it only awares whether there is still available resource, but does not care the amount of available resource.
    the amount of available resource is handled by JIT engine (i.e., NasAdvisor)
    """
    try:
        _logger.info('start main')
        best_graph = sdk.experiment.base_model
        _logger.info('get graph object, start mutator class')
        mutators = sdk.experiment.mutators

        _logger.info('start sampler')
        sampler = RandomSampler()

        for _ in range(5):
            _logger.info('start main loop')
            graphs = []
            for _ in range(3):
                new_graph = best_graph.duplicate()
                for mutator in mutators:
                    # Note: must use returned graph
                    new_graph = mutator.apply(new_graph, sampler)
                graphs.append(new_graph)
            _logger.info('mutated graph')
            sdk.train_graphs(graphs)
            _logger.info('train graph')
            print([graph.metrics for graph in graphs])
            best_graph = sorted(graphs, key=(lambda graph: graph.metrics))[-1]
    except Exception as e:
        # has to catch error here, because main thread cannot show the error in this thread
        _logger.error(logging.exception('message'))


class RandomSampler(sdk.strategy.Sampler):
    def choice(self, candidates):
        return random.choice(candidates)
