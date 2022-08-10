# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import graphviz


def convert_to_visualize(graph_ir, vgraph):
    for name, graph in graph_ir.items():
        if name == '_evaluator':
            continue
        with vgraph.subgraph(name='cluster'+name) as subgraph:
            subgraph.attr(color='blue')
            cell_node = {}
            ioput = {'_inputs': '{}-{}'.format(name, '_'.join(graph['inputs'])),
                     '_outputs': '{}-{}'.format(name, '_'.join(graph['outputs']))}
            subgraph.node(ioput['_inputs'])
            subgraph.node(ioput['_outputs'])
            for node_name, node_value in graph['nodes'].items():
                value = node_value['operation']
                if value['type'] == '_cell':
                    cell_input_name = '{}-{}'.format(value['cell_name'], '_'.join(graph_ir[value['cell_name']]['inputs']))
                    cell_output_name = '{}-{}'.format(value['cell_name'], '_'.join(graph_ir[value['cell_name']]['outputs']))
                    cell_node[node_name] = (cell_input_name, cell_output_name)
                    print('cell: ', node_name, cell_input_name, cell_output_name)
                else:
                    subgraph.node(node_name)
            for edge in graph['edges']:
                src = edge['head'][0]
                if src == '_inputs':
                    src = ioput['_inputs']
                elif src in cell_node:
                    src = cell_node[src][1]
                dst = edge['tail'][0]
                if dst == '_outputs':
                    dst = ioput['_outputs']
                elif dst in cell_node:
                    dst = cell_node[dst][0]
                subgraph.edge(src, dst)


def visualize_model(graph_ir):
    vgraph = graphviz.Digraph('G', filename='vgraph', format='jpg')
    convert_to_visualize(graph_ir, vgraph)
    vgraph.render()
