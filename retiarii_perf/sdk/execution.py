import time
import logging

from . import nni_integration

_logger = logging.getLogger(__name__)

def submit_graph(graph: 'Graph') -> None:
    #nni_integration.get_tuner().enqueue(graph)
    nni_integration.get_advisor().enqueue(graph)
    _logger.info('submit one graph')

def submit_graphs(graphs: 'Iterable[Graph]') -> None:
    nni_integration.get_advisor().enqueue(graphs)
    _logger.info('submit {} graphs'.format(len(graphs)))

def wait_graph(graph: 'Graph') -> None:
    while graph.metrics is None:
        time.sleep(1)

def wait_graphs(graphs: 'Iterable[Graph]') -> None:
    _logger.info('start waiting {} graphs'.format(len(graphs)))
    for graph in graphs:
        _logger.info('wait addr: {}'.format(graph))
    while any(graph.metrics is None for graph in graphs):
        time.sleep(1)

def train_graph(graph: 'Graph') -> 'Any':
    submit_graph(graph)
    wait_graph(graph)
    return graph.metrics

def train_graphs(graphs: 'Iterable[Graph]') -> 'Any':
    _logger.info('start train graph')
    submit_graphs(graphs)
    _logger.info('submit graphs')
    wait_graphs(graphs)
    _logger.info('wait graphs')
