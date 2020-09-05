import random
import logging
import sdk

_logger = logging.getLogger(__name__)

class GraphChoice:
    def __init__(self, graph, choices):
        self.graph = graph
        self.choices = choices

def main():
    """
    this one can be seen as a scheduler in strategy, and there is another component which decides the choices.
    for simplicity, it only awares whether there is still available resource, but does not care the amount of available resource.
    the amount of available resource is handled by JIT engine (i.e., NasAdvisor)
    """
    try:
        iteration_num = 100
        population_size = 3
        top_num = 1
        top_trials = []
        _logger.info('start main')
        base_graph = sdk.experiment.base_model
        _logger.info('get graph object, start mutator class')
        mutators = sdk.experiment.mutators

        _logger.info('start...')
        for i in range(iteration_num):
            _logger.info('iteration {}'.format(i))
            graphs = []
            graphs_choice = []
            for _ in range(population_size):
                new_graph = base_graph.duplicate()
                if top_trials:
                    choice = random.choice(top_trials)
                    sampler = MutatorSampler(choice)
                else:
                    sampler = MutatorSampler()
                for mutator in mutators:
                    # Note: must use returned graph
                    new_graph = mutator.apply(new_graph, sampler)
                graphs.append(new_graph)
                graphs_choice.append(GraphChoice(new_graph, sampler.get_choices()))
            _logger.info('mutated graph')
            sdk.train_graphs(graphs)
            _logger.info('train graph')
            print([graph.metrics for graph in graphs])
            sorted_graph = sorted(graphs_choice, key=(lambda graph_choice: graph_choice.graph.metrics))
            top_trials = []
            for j in range(1, top_num+1):
                top_trials.append(sorted_graph[-j].choices)
    except Exception as e:
        # has to catch error here, because main thread cannot show the error in this thread
        _logger.error(logging.exception('message'))


class MutatorSampler(sdk.strategy.Sampler):
    def __init__(self, choices=None):
        if choices is not None:
            self.from_mutate = True
            self.choices = choices
            # randomly mutate one
            idx = random.randint(0, len(choices)-1)
            self.choices[idx] = None
            self.idx = 0
        else:
            self.from_mutate = False
            self.choices = []

    def choice(self, candidates):
        if self.from_mutate == True:
            if self.choices[self.idx] is None:
                choice = random.choice(candidates)
                self.choices[self.idx] = choice
                self.idx += 1
                return choice
            else:
                choice = self.choices[self.idx]
                self.idx += 1
                return choice
        else:
            choice = random.choice(candidates)
            self.choices.append(choice)
            return choice

    def get_choices(self):
        return self.choices
