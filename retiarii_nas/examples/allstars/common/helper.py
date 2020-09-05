from sdk.graph import Graph
from sdk.utils import import_


def graph_to_module_instance(graph, dependencies):
    graph = Graph.load(graph)
    graph.generate_code('pytorch', output_file=f'generated/graph_tmp.py')

    # hack: add import part
    with open(f'generated/graph_tmp.py') as f:
        code = f.read()
    code = ''.join([f'from {d} import *\n' for d in dependencies]) + code
    with open(f'generated/graph_tmp.py', 'w') as f:
        f.write(code)

    model_cls = import_(f'generated.graph_tmp.Graph')
    model = model_cls()
    return model
