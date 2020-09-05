import graphviz

def convert_to_visualize(json_graph):
    vgraph = graphviz.Digraph(format='png')
    graph = json_graph['graph']
    render_cfg = {'shape': 'ellipse', 'style': 'solid'}
    # handle inputs
    inputs = graph['inputs']
    for _input in inputs:
        vgraph.node(_input['name'], **render_cfg)
    # handle outputs
    outputs = graph['outputs']
    for _output in outputs:
        vgraph.node(_output['name'], **render_cfg)
    # handle hidden nodes
    hnodes = graph['hidden_nodes']
    for hnode in hnodes:
        vgraph.node(hnode['name'], hnode['operation']['type'], **render_cfg)
    # handle edges
    edges = graph['edges']
    for edge in edges:
        vgraph.edge(edge['head'], edge['tail'])
    vgraph.render('vgraph')
    return vgraph
