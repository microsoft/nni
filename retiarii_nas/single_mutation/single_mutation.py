import sdk


class AddBatchNorm(sdk.Mutator):
    def mutate(self, graph):
        conv_edges = []
        for edge in graph.edges:
            if edge.head.operation and edge.head.operation.type.startswith('conv'):
                conv_edges.append(edge)

        for edge in conv_edges:
            batch_norm = graph.add_node('batch_normalization')
            graph.add_edge(edge.head, batch_norm)
            graph.add_edge(batch_norm, edge.tail)
            graph.remove_edge(edge)


def main():
    orig_graph = sdk.experiment.base_graph
    new_graph = AddBatchNorm(orig_graph).apply([])
    sdk.train_graph(new_graph)
