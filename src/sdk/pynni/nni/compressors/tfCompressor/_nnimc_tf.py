import tensorflow as tf
from tensorflow import Graph, Operation, Tensor
import yaml
from typing import List

__all__ = [
    'TfCompressor',
    'TfPruner',
    'TfQuantizer',
    '_tf_detect_prunable_layers',
    '_tf_default_get_configure',
    '_tf_default_load_configure_file'
]


class TfCompressor:
    """TODO"""

    def __init__(self):
        self._bound_model = None


    def compress(self, model):
        """
        Compress given graph with algorithm implemented by subclass.
        This will edit the graph.
        """
        assert self._bound_model is None, "Each NNI compressor instance can only compress one model"
        self._bound_model = model
        self.preprocess_model(model)

    def compress_default_graph(self):
        """
        Compress the default graph with algorithm implemented by subclass.
        This will edit the graph.
        """
        self.compress(tf.get_default_graph())


    def preprocess_model(self, model):
        """
        This method is called when a model is bound to the compressor.
        Users can optionally overload this method to do model-specific initialization.
        It is guaranteed that only one model will be bound to each compressor instance.
        """
        pass


class TfLayerInfo:
    def __init__(self, layer):
        self.name = layer.name
        self.layer = layer
        self.weight_index = None

        self.support_type = ['Conv2D', 'MatMul', 'DepthwiseConv2dNative']
        if layer.type in self.support_type:
            self.weight_index = 1
        else:
            raise ValueError('Unsupported layer')


def _tf_detect_prunable_layers(model):
    # search for Conv2D layers
    # TODO: whitelist
    whiltlist = ['Conv2D', 'MatMul', 'DepthwiseConv2dNative']
    return [ TfLayerInfo(op) for op in model.get_operations() if op.type in whiltlist ]

def _tf_default_get_configure(configure_list, layer_info):
    configure = {}
    for config in configure_list:
        if config.get('support_type', '') == 'default':
            configure = config
        elif layer_info.layer.type in config.get('support_type', []):
            configure = config
        elif layer_info.name in config.get('support_op', []):
            configure = config

    return configure

def _tf_default_load_configure_file(config_path, class_name):
    assert config_path is not None and config_path.endswith('yaml')
    file = open(config_path, 'r')
    yaml_text = yaml.load(file.read())
    return yaml_text.get(class_name, {})

class TfPruner(TfCompressor):
    """TODO"""

    def __init__(self):
        super().__init__()

    def __call__(self, model):
        self.compress(model)
        return model

    def calc_mask(self, layer_info, weight):
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `multiply()` operation.
        This method works as a subgraph which will be inserted into the bound model.
        """
        raise NotImplementedError("Pruners must overload calc_mask()")

    def compress(self, model):
        super().compress(model)
        # TODO: configurable whitelist
        for layer_info in _tf_detect_prunable_layers(model):
            self._instrument_layer(layer_info)

    def _instrument_layer(self, layer_info):
        # it seems the graph editor can only swap edges of nodes or remove all edges from a node
        # it cannot remove one edge from a node, nor can it assign a new edge to a node
        # we assume there is a proxy operation between the weight and the Conv2D layer
        # this is true as long as the weight is `tf.Value`
        # not sure what will happen if the weight is calculated from other operations
        weight_op = layer_info.layer.inputs[layer_info.weight_index].op
        weight = weight_op.inputs[0]
        mask = self.calc_mask(layer_info, weight)
        new_weight = weight * mask
        tf.contrib.graph_editor.swap_outputs(weight_op, new_weight.op)
    
    def update_epoch(self, epoch, sess):
        pass
    
    def step(self, sess):
        pass


class TfQuantizer(TfCompressor):
    def __init__(self):
        super().__init__()

    def __call__(self, model):
        self.compress(model)
        return model

    def quantize_weight(self, layer_info, weight):
        raise NotImplementedError()


    def compress(self,  model):
        for layer_info in _tf_detect_prunable_layers(model):
            self._instrument_layer(layer_info)

    def _instrument_layer(self, layer_info):
        weight_op = layer_info.layer.inputs[layer_info.weight_index].op
        new_weight = self.quantize_weight(layer_info, weight_op.inputs[0])
        tf.contrib.graph_editor.swap_outputs(weight_op, new_weight.op)

    def update_epoch(self, epoch, sess):
        pass
    
    def step(self, sess):
        pass