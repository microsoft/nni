import tensorflow.compat.v1 as tf
import logging
from . import default_layers

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, op):
        self.op = op
        self.name = op.name
        self.type = op.type


class Compressor:
    """Abstract base TensorFlow compressor"""

    def __init__(self, config_list):
        self._bound_model = None
        self._config_list = config_list

    def __call__(self, model):
        """Compress given graph with algorithm implemented by subclass.
        The graph will be editted and returned.
        """
        self.compress(model)
        return model

    def compress(self, model):
        """Compress given graph with algorithm implemented by subclass.
        This will edit the graph.
        """
        assert self._bound_model is None, "Each NNI compressor instance can only compress one model"
        self._bound_model = model
        self.bind_model(model)
        for op in model.get_operations():
            layer = LayerInfo(op)
            config = self._select_config(layer)
            if config is not None:
                self._instrument_layer(layer, config)

    def compress_default_graph(self):
        """Compress the default graph with algorithm implemented by subclass.
        This will edit the default graph.
        """
        self.compress(tf.get_default_graph())


    def bind_model(self, model):
        """This method is called when a model is bound to the compressor.
        Compressors can optionally overload this method to do model-specific initialization.
        It is guaranteed that only one model will be bound to each compressor instance.
        """
        pass
    
    def update_epoch(self, epoch, sess):
        """If user want to update mask every epoch, user can override this method
        """
        pass
    
    def step(self, sess):
        """If user want to update mask every step, user can override this method
        """
        pass


    def _instrument_layer(self, layer, config):
        raise NotImplementedError()

    def _select_config(self, layer):
        ret = None
        for config in self._config_list:
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


class Pruner(Compressor):
    """Abstract base TensorFlow pruner"""

    def __init__(self, config_list):
        super().__init__(config_list)

    def calc_mask(self, weight, config, op, op_type, op_name):
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
        weight_index = _detect_weight_index(layer)
        if weight_index is None:
            _logger.warning('Failed to detect weight for layer {}'.format(layer.name))
            return
        weight_op = layer.op.inputs[weight_index].op
        weight = weight_op.inputs[0]
        mask = self.calc_mask(weight, config, op=layer.op, op_type=layer.type, op_name=layer.name)
        new_weight = weight * mask
        tf.contrib.graph_editor.swap_outputs(weight_op, new_weight.op)


class Quantizer(Compressor):
    """Abstract base TensorFlow quantizer"""

    def __init__(self, config_list):
        super().__init__(config_list)

    def quantize_weight(self, weight, config, op, op_type, op_name):
        raise NotImplementedError("Quantizer must overload quantize_weight()")

    def _instrument_layer(self, layer, config):
        weight_index = _detect_weight_index(layer)
        if weight_index is None:
            _logger.warning('Failed to detect weight for layer {}'.format(layer.name))
            return
        weight_op = layer.op.inputs[weight_index].op
        weight = weight_op.inputs[0]
        new_weight = self.quantize_weight(weight, config, op=layer.op, op_type=layer.type, op_name=layer.name)
        tf.contrib.graph_editor.swap_outputs(weight_op, new_weight.op)


def _detect_weight_index(layer):
    index = default_layers.op_weight_index.get(layer.type)
    if index is not None:
        return index
    weight_indices = [ i for i, op in enumerate(layer.op.inputs) if op.name.endswith('Variable/read') ]
    if len(weight_indices) == 1:
        return weight_indices[0]
    return None
