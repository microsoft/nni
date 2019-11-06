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
    """
    Abstract base TensorFlow compressor
    """

    def __init__(self, model, config_list):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        config_list : list
            the configurations that users specify for compression
        """
        self.bound_model = model
        self.config_list = config_list
        self.modules_to_compress = None

    def detect_modules_to_compress(self):
        """
        detect all modules should be compressed, and save the result in `self.modules_to_compress`.

        The model will be instrumented and user should never edit it after calling this method.
        """
        if self.modules_to_compress is None:
            self.modules_to_compress = []
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
                    self.modules_to_compress.append((layer, config))
        return self.modules_to_compress
    
    def compress(self):
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self.modules_to_compress` records all the to-be-compressed layers
        """
        modules_to_compress = self.detect_modules_to_compress()
        for layer, config in modules_to_compress:
            self._instrument_layer(layer, config)
        return self.bound_model

    def get_modules_to_compress(self):
        """
        To obtain all the to-be-compressed layers.

        Returns
        -------
        self.modules_to_compress : list
            a list of the layers, each of which is a tuple (`layer`, `config`),
            `layer` is `LayerInfo`, `config` is a `dict`
        """
        return self.modules_to_compress

    def select_config(self, layer):
        """
        Find the configuration for `layer` by parsing `self.config_list`

        Parameters
        ----------
        layer : LayerInfo
            one layer

        Returns
        -------
        ret : config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
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
        """
        If user want to update model every epoch, user can override this method.
        This method should be called at the beginning of each epoch

        Parameters
        ----------
        epoch : num
            the current epoch number
        """

    def step(self, sess):
        """
        If user want to update mask every step, user can override this method
        """


    def _instrument_layer(self, layer, config):
        """
        This method is implemented in the subclasses, i.e., `Pruner` and `Quantizer`

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            the configuration for compressing this layer
        """
        raise NotImplementedError()


class Pruner(Compressor):
    """
    Abstract base TensorFlow pruner
    """

    def calc_mask(self, layer, config):
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `mul()` operation on the weight.
        This method is effectively hooked to `forward()` method of the model.

        Parameters
        ----------
        layer : LayerInfo
            calculate mask for `layer`'s weight
        config : dict
            the configuration for generating the mask
        """
        raise NotImplementedError("Pruners must overload calc_mask()")

    def _instrument_layer(self, layer, config):
        """
        Create a wrapper forward function to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for generating the mask
        """
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
