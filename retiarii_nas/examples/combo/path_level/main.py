import random
from sdk import *
from sdk.operations_tf import Operation


ops = [
    Operation.new(type='Identity'),
    Operation.new(type='Conv2D', filters=96, kernel_size=1, activation='relu'),
    Operation.new(type='DepthwiseConv2D', kernel_size=3, activation='relu'),
    Operation.new(type='DepthwiseConv2D', kernel_size=5, activation='relu'),
    Operation.new(type='DepthwiseConv2D', kernel_size=7, activation='relu'),
    Operation.new(type='PathLevel__avg_pool'),
    Operation.new(type='PathLevel__max_pool'),
]

identity = ops[0]



class TreeMotif:
    def __init__(self, merge='concat', leaf=identity):
        self.merge = merge
        self.children = [leaf, identity]

    def find_leaves(self):
        ret = []
        for i, child in enumerate(self.children):
            if isinstance(child, Operation):
                ret.append((self, i))
            else:
                ret += child.find_leaves()
        return ret


class PathLevelMutator(Mutator):
    def __init__(self, tree):
        super().__init__()
        self.tree = tree

    def mutate(self, graph):
        leaf_parent, leaf_idx = self.choice(self.tree.find_leaves())
        action = self.choice([self.split, self.replace, self.replace])
        action(leaf_parent, leaf_idx)
        _mutate(graph, (graph.input_node, 0), graph.output_node, self.tree)

    def split(self, leaf_parent, leaf_idx):
        merge = self.choice(['add', 'concat'])
        leaf = leaf_parent.children[leaf_idx]
        leaf_parent.children[leaf_idx] = TreeMotif(merge, leaf)

    def replace(self, leaf_parent, leaf_idx):
        leaf_parent.children[leaf_idx] = self.choice(ops)


def _mutate(graph, head, tail, tree):
    if isinstance(tree, Operation):  # leaf:
        node = graph.add_node(tree)
        graph.add_edge(head, node)
        graph.add_edge(node, tail)
        return

    if tree.merge == 'sum':
        divide_op = Operation.new(type='PathLevel__split')
        unite_op = Operation.new(type='Sum')
    else:
        divide_op = Operation.new(type='Replication')
        unite_op = Operation.new(type='Concatenate', dimension=3)

    divide = graph.add_node(divide_op)
    unite = graph.add_node(unite_op)
    graph.add_edge(head, divide)
    graph.add_edge(unite, tail)

    for i, children in enumerate(tree.children):
        _mutate(graph, (divide, i), unite, tree.children[i])



class RandomSampler(Sampler):
    def choice(self, candidates):
        return random.choice(candidates)


def main():
    empty_graph = experiment.base_graph.cell_templates['Net']
    tree = TreeMotif()
    for _ in range(1000):
        graph = PathLevelMutator(tree).apply(empty_graph, RandomSampler())
        train_graph(graph)
