import logging

_logger = logging.getLogger(__name__)

class OptGraph:
    def __init__(self, opt_graph, resource, graphs):
        """
        Parameters
        ----------
        opt_graph : Graph
            the merged graph
        resource : dict
            allocated resource to opt_graph
        graphs : list of Graph
            list of graphs that are merged in opt_graph
        """
        self.opt_graph = opt_graph
        self.resource = resource
        assert isinstance(graphs, list)
        self.graphs = graphs

def optimize_graphs(graphs, avail_resource):
    """
    Parameters
    ----------
    graphs : list of Graph
        list of graphs to optimize together
    avail_resource : int
        available resource

    Returns
    -------
    list
        list of OptGraph
    list
        list of left graphs
    """
    # for now, we simply allocate resource and generate code for each graph
    opt_graphs = []
    left_graphs = None
    for i, graph in enumerate(graphs):
        if avail_resource == 0:
            left_graphs = graphs[i:]
            break
        opt_graph = graph.duplicate()
        opt_graphs.append(OptGraph(opt_graph, 1, [graph]))
        avail_resource -= 1
        _logger.info('obj addr: {} {}'.format(graphs[i], opt_graphs[-1].graphs[0]))
        assert graphs[i] == opt_graphs[-1].graphs[0]
    return opt_graphs, left_graphs
