import logging
import tensorflow as tf
from . import default_layers

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, op, weight, weight_op):
        self.op = op
        self.name = op.name
        self.type = op.type
        self.weight = weight
        self.weight_op = weight_op


class Compressor:
    """Abstract base TensorFlow compressor"""

    def __init__(self, model, config_list):
        self.bound_model = model
        self.config_list = config_list
        self.modules_to_compress = []

    def compress(self):
        """Compress given graph with algorithm implemented by subclass.
        This will edit the graph.
        """
        for op in self.bound_model.get_operations():
            weight_index = _detect_weight_index(op)
            if weight_index is None:
                _logger.warning('Failed to detect weight for layer %s', op.name)
                return
            weight_op = op.inputs[weight_index].op
            weight = weight_op.inputs[0]

            layer = LayerInfo(op, weight, weight_op)
            config = self.select_config(layer)
            if config is not None:
                self._instrument_layer(layer, config)
                self.modules_to_compress.append((layer, config))
        return self.bound_model

    def get_modules_to_compress(self):
        return self.modules_to_compress

    def select_config(self, layer):
        ret = None
        for config in self.config_list:
            op_types = config.get('op_types')
            if op_types == 'default':
                op_types = default_layers.op_weight_index.keys()
            if op_types and layer.type not in op_types:
                continue
            if config.get('op_names') and layer.name not in config['op_names']:
                continue
            ret = config
        if ret is None or ret.get('exclude'):
            return None
        return ret

    def update_epoch(self, epoch, sess):
        """If user want to update mask every epoch, user can override this method
        """

    def step(self, sess):
        """If user want to update mask every step, user can override this method
        """


    def _instrument_layer(self, layer, config):
        raise NotImplementedError()


class Pruner(Compressor):
    """
    Abstract base TensorFlow pruner
    """

    def calc_mask(self, layer, config):
        """Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `multiply()` operation.
        This method works as a subgraph which will be inserted into the bound model.
        """
        raise NotImplementedError("Pruners must overload calc_mask()")

    def _instrument_layer(self, layer, config):
        # it seems the graph editor can only swap edges of nodes or remove all edges from a node
        # it cannot remove one edge from a node, nor can it assign a new edge to a node
        # we assume there is a proxy operation between the weight and the Conv2D layer
        # this is true as long as the weight is `tf.Value`
        # not sure what will happen if the weight is calculated from other operations
        mask = self.calc_mask(layer, config)
        new_weight = layer.weight * mask
        tf.contrib.graph_editor.swap_outputs(layer.weight_op, new_weight.op)


class Quantizer(Compressor):
    """
    Abstract base TensorFlow quantizer
    """

    def quantize_weight(self, weight, config, op, op_type, op_name):
        raise NotImplementedError("Quantizer must overload quantize_weight()")

    def _instrument_layer(self, layer, config):
        weight_index = _detect_weight_index(layer)
        if weight_index is None:
            _logger.warning('Failed to detect weight for layer %s', layer.name)
            return
        weight_op = layer.op.inputs[weight_index].op
        weight = weight_op.inputs[0]
        new_weight = self.quantize_weight(weight, config, op=layer.op, op_type=layer.type, op_name=layer.name)
        tf.contrib.graph_editor.swap_outputs(weight_op, new_weight.op)


def _detect_weight_index(layer):
    index = default_layers.op_weight_index.get(layer.type)
    if index is not None:
        return index
    weight_indices = [i for i, op in enumerate(layer.inputs) if op.name.endswith('Variable/read')]
    if len(weight_indices) == 1:
        return weight_indices[0]
    return None
