from torch import Tensor
from torch.nn import Module, Parameter

from typing import List

__all__ = [
    'TorchCompressor',
    'TorchPruner',
    'TorchQuantizer',
    '_torch_detect_prunable_layers'
]


class TorchCompressor:
    """TODO"""

    def __init__(self) -> None:
        self._bound_model: Optional[Module] = None


    def compress(self, model: Module) -> None:
        """
        Compress the model with algorithm implemented by subclass.
        The model will be instrumented and user should never edit it after calling this method.
        """
        assert self._bound_model is None, "Each NNI compressor instance can only compress one model"
        self._bound_model = model
        self.bind_model(model)


    def bind_model(self, model: Module) -> None:
        """
        This method is called when a model is bound to the compressor.
        Users can optionally overload this method to do model-specific initialization.
        It is guaranteed that only one model will be bound to each compressor instance.
        """
        pass


class TorchLayerInfo:
    def __init__(self, name: str, layer: Module):
        self.name: str = name
        self.layer: Module = layer

        self._forward: Optional[Function] = None


def _torch_detect_prunable_layers(model: Module) -> List[TorchLayerInfo]:
    # search for all layers which have parameter "weight"
    ret = [ ]
    for name, layer in model.named_modules():
        try:
            if isinstance(layer.weight, Parameter) and isinstance(layer.weight.data, Tensor):
                ret.append(TorchLayerInfo(name, layer))
        except AttributeError:
            pass
    return ret


class TorchPruner(TorchCompressor):
    """TODO"""

    def __init__(self) -> None:
        super().__init__()

    def calc_mask(self, layer_info: TorchLayerInfo, weight: Tensor) -> Tensor:
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `mul()` operation.
        This method is effectively hooked to `forward()` method of the model.
        """
        raise NotImplementedError("Pruners must overload calc_mask()")


    def compress(self, model: Module) -> None:
        super().compress(model)
        # TODO: configurable whitelist
        for layer_info in _torch_detect_prunable_layers(model):
            self._instrument_layer(layer_info)

    def _instrument_layer(self, layer_info: TorchLayerInfo):
        # TODO: bind additional properties to layer_info instead of layer
        # create a wrapper forward function to replace the original one
        assert layer_info._forward is None, 'Each model can only be compressed once'
        layer_info._forward = layer_info.layer.forward

        def new_forward(*input):
            # apply mask to weight
            mask = self.calc_mask(layer_info, layer_info.layer.weight.data)
            layer_info._backup_weight = layer_info.layer.weight.data
            layer_info.layer.weight.data = layer_info.layer.weight.data.mul(mask)
            # calculate forward
            ret = layer_info._forward(*input)
            # recover original weight
            layer_info.layer.weight.data = layer_info._backup_weight
            return ret

        layer_info.layer.forward = new_forward


class TorchQuantizer(TorchCompressor):
    def __init__(self) -> None:
        super().__init__()

    def quantize_weight(self, layer_info: TorchLayerInfo, weight: Tensor) -> Tensor:
        # FIXME: where dequantize goes?
        raise NotImplementedError()

    def compress(self, model: Module) -> None:
        super().compress(model)
        count = 0
        for layer_info in _torch_detect_prunable_layers(model):
            if count == 0:
                count = count +1
                continue
            self._instrument_layer(layer_info)

    def _instrument_layer(self, layer_info: TorchLayerInfo):
        assert layer_info._forward is None
        layer_info._forward = layer_info.layer.forward

        def new_forward(*input):
            layer_info.layer.weight.data = self.quantize_weight(layer_info, layer_info.layer.weight.data)
            return layer_info._forward(*input)

        layer_info.layer.forward = new_forward
