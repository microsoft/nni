import sdk

class ExampleMutator(sdk.Mutator):
    def mutate(self, graph):
        nodes = graph.find_nodes_by_type('Conv2d')
        node = self.choice(nodes)
        op = self.choice(sdk.experiment.hyper_parameters['operations'])
        node.update_operation('Conv2d', **op)

        '''node = self.choice(graph.hidden_nodes)
        op = self.choice(sdk.experiment.hyper_parameters['operations'])
        node.set_operation(**op)

        skip_nodes = graph.hidden_nodes + graph.output_nodes
        n = len(skip_nodes)
        head_idx = self.choice(range(0, n - 2))
        tail_idx = self.choice(range(head_idx + 2, n))
        graph.add_edge(skip_nodes[head_idx], skip_nodes[tail_idx])'''
