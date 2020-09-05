import random
from sdk import *


M = [6, 6, 1]
L = 3
G_node_num = [None, 4, 5]

initial_genotypes_num = 1000
tournament_size = 0.5
total_steps = 1000


class HierarchicalMutator(Mutator):
    def mutate(self, graph):
        l = self.choice(range(1, L))
        m = self.choice(range(M[l]))
        i = self.choice(range(G_node_num[l] -  1))
        j = self.choice(range(i + 1, G_node_num[l]))
        k = self.choice(range(M[l - 1] + 1))

        motif = graph.cell_templates['Motif_{}_{}'.format(l, m)]
        node = motif.find_node('node_{}_{}'.format(i, j))
        if k < M[l - 1]:
            op = graph.cell_templates['Motif_{}_{}'.format(l - 1, k)]
        else:
            op = graph.cell_templates['Motif_none']
        node.set_template(op)


class RandomSampler(Sampler):
    def choice(self, candidates):
        return random.choice(candidates)

def _mutate(graph):
    return HierarchicalMutator().apply(graph, RandomSampler())


def main():
    trivial = experiment.base_graph

    genotypes = [_mutate(trivial) for _ in range(initial_genotypes_num)]
    train_graphs(genotypes)

    for _ in range(total_steps):
        k = int(len(genotypes) * tournament_size)
        individuals = random.sample(genotypes, k)
        winner = individuals[0]
        for individual in individuals:
            if individual.metrics['accuracy'] > winner.metrics['accuracy']:
                winner = individual
        new = _mutate(winner)
        train_graph(new)
        genotypes.append(new)


if __name__ == '__main__':
    main()
