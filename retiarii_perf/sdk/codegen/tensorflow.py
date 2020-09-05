def graph_to_tensorflow_script(graph: 'Graph') -> str:
    return _TensorFlowScriptTemplate.format(graph_to_tensorflow_model(graph)).strip()

def graph_to_tensorflow_model(graph: 'Graph') -> str:
    nodes = graph.topo_sort()

    node_codes = ['self.{} = {}'.format(node.name, node.operation.to_tensorflow_init()) for node in nodes]

    edge_codes = []

    for node in nodes:
        predecessors = graph.get_predecessors(node)
        if not predecessors:
            inputs = ''
        elif len(predecessors) == 1:
            inputs = predecessors[0].name
        else:
            inputs = 'tf.add_n([{}])'.format(', '.join(node.name for node in predecessors))
        edge_codes.append('{} = self.{}({})'.format(node.name, node.name, inputs))

    for node in graph.output_nodes:
        predecessors = graph.get_predecessors(node)
        if not predecessors:
            value = 'None'
        elif len(predecessors) == 1:
            value = predecessors[0].name
        else:
            value = 'tf.add_n([{}])'.format(', '.join(node.name for node in predecessors))
        edge_codes.append('{} = {}'.format(node.name, value))

    try:
        input_code = ', '.join(node.name for node in graph.input_nodes)
        output_code = ', '.join(node.name for node in graph.output_nodes)
    except AttributeError:
        # to be compatible with the new format
        input_code = ', '.join(node.name for node in graph.input_nodes[0])
        output_code = ', '.join(node.name for node in graph.output_nodes[0])

    linebreak = '\n        '
    return _TensorFlowModelTemplate.format(
        graph_name='Graph',
        inputs=input_code,
        outputs=output_code,
        nodes=linebreak.join(node_codes),
        edges=linebreak.join(edge_codes)
    )


_TensorFlowScriptTemplate = '''
import tensorflow as tf
import tensorflow.keras as K

import sdk.custom_ops_tf as CUSTOM

{}
'''

_TensorFlowModelTemplate = '''
class {graph_name}(K.Model):
    def __init__(self):
        super().__init__()
        {nodes}

    def call(self, {inputs}):
        {edges}
        return {outputs}
'''
