import math
import random
from sdk import *


activations = [
    'linear',
    # 'inverse',
    'tanh',
    'sigmoid',
    'relu',
    # 'step',
    # 'sine',
    # 'cosine',
    # 'gaussian',
    # 'absolute',
]


population_size = 960
generations = 4096
initial_active = 0.05
tournament_size = 32

tournament_prob = 0.8


class WannMutator(Mutator):
    def mutate(self, graph):
        action = self.choice([
            self.insert_node,
            self.add_connection,
            self.change_activation,
        ])
        action(graph)


    def insert_node(self, graph):
        edges = graph.edges[10:]
        if not edges:
            return

        edge = self.choice(edges)
        activ = self.choice(activations)

        head = (edge.head, edge.head_idx)
        tail = (edge.tail, edge.tail_idx)
        node = graph.add_node(type="Wann__activation", activation=activ)

        graph.add_edge(head, node)
        graph.add_edge(node, edge)
        graph.remove_edge(edge)


    def change_activation(self, graph):
        hidden = graph.hidden_nodes[10:]
        if not hidden:
            return

        node = self.choice(hidden)
        activ = self.choice(activations)
        node.update_operation(type="Wann__activation", activation=activ)


    def add_connection(self, graph):
        outputs = graph.hidden_nodes[:10]
        hidden = graph.hidden_nodes[10:]

        head_idx = self.choice(range(len(hidden) + 196))
        if head_idx < len(hidden):  # choose hidden node
            head = (hidden[head_idx], None)
            tail_start = head_idx + 1
        else:  # choose input node
            head = (graph.input_node, head_idx - len(hidden))
            tail_start = 0

        tail_idx = self.choice(range(tail_start, len(hidden) + 10))
        if tail_idx < len(hidden):  # choose hidden node
            tail = (hidden[tail_idx], None)
        else:  # choose output node
            tail = (outputs[tail_idx - len(hidden)], None)

        for edge in graph.edges:
            if edge == (head[0], head[1], tail[0], tail[1]):
                return
        graph.add_edge(head, tail)


class WannInitMutator(Mutator):
    def mutate(self, graph):
        for input_idx in range(196):
            for output_idx in range(10):
                if random.random() < initial_active:
                    out = (graph.find_node('out_{}'.format(output_idx)), None)
                    graph.add_edge((graph.input_node, input_idx), out)


class RandomSampler(Sampler):
    def choice(self, candidates):
        return random.choice(candidates)


def _mutate(MutatorClass, graph):
    return MutatorClass().apply(graph, RandomSampler())

def _rank_graph(graph):
    return -(graph.metrics['accuracy'] / math.log(len(graph.hidden_nodes)))


def main():
    graph = experiment.base_graph.cell_templates['WannNet']

    population = [_mutate(WannInitMutator, graph) for _ in range(population_size)]
    train_graphs(population)

    for _ in range(generations):
        new_population = []
        while len(new_population) < population_size:
            candidates = random.sample(population, tournament_size)
            candidates = sorted(candidates, key=(lambda graph: _rank_graph(graph)))
            prob = tournament_prob
            for graph in candidates:
                if random.random() < prob:
                    new_population.append(_mutate(WannMutator, graph))
                prob *= (1 - tournament_prob)

        population = new_population
        train_graphs(population)
