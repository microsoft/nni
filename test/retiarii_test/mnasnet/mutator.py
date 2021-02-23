import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from nni.retiarii import Mutator

from base_mnasnet import RegularConv, DepthwiseConv, MobileConv

_logger = logging.getLogger(__name__)

class BlockMutator(Mutator):
    def __init__(self, target: str):
        super(BlockMutator, self).__init__()
        self.target = target

    def mutate(self, model):
        nodes = model.get_nodes_by_label(self.target)
        assert len(nodes) == 1
        node = nodes[0]
        graph = node.graph

        related_info = node.operation.parameters
        kernel_size = self.choice(related_info['kernel_size_options'])
        op_type = self.choice(related_info['op_type_options'])
        #self.choice(related_info['se_ratio_options'])
        skip = self.choice(related_info['skip_options'])
        n_filter = self.choice(related_info['n_filter_options'])

        if related_info['in_ch'] is not None:
            in_ch = related_info['in_ch']
        else:
            assert len(node.predecessors) == 1
            the_node = node.predecessors[0]
            _logger.debug(repr(the_node.operation.parameters))
            _logger.debug(the_node.__repr__())
            in_ch = the_node.operation.parameters['out_ch']

        # update the placeholder to be a new operation
        node.update_operation(op_type, {
            'kernel_size': kernel_size,
            'in_ch': in_ch,
            'out_ch': n_filter,
            'skip': 'no',
            'exp_ratio': related_info['exp_ratio'],
            'stride': related_info['stride']
        })

        # insert new nodes after the placeholder
        n_layer = self.choice(related_info['n_layer_options'])
        for i in range(1, n_layer):
            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                                             '{}_{}'.format(self.target, i),
                                             op_type,
                                             {'kernel_size': kernel_size,
                                              'in_ch': n_filter,
                                              'out_ch': n_filter,
                                              'skip': skip,
                                              'exp_ratio': related_info['exp_ratio'],
                                              'stride': 1})

        # fix possible shape mismatch
        # TODO: use formal method function to update parameters
        if len(node.successors) == 1 and 'in_channels' in node.successors[0].operation.parameters:
            node.successors[0].operation.parameters['in_channels'] = n_filter