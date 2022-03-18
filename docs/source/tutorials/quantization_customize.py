"""
Customize a new quantization algorithm
======================================

To write a new quantization algorithm, you can write a class that inherits ``nni.compression.pytorch.Quantizer``.
Then, override the member functions with the logic of your algorithm. The member function to override is ``quantize_weight``.
``quantize_weight`` directly returns the quantized weights rather than mask, because for quantization the quantized weights cannot be obtained by applying mask.
"""

from nni.compression.pytorch import Quantizer

class YourQuantizer(Quantizer):
    def __init__(self, model, config_list):
        """
        Suggest you to use the NNI defined spec for config
        """
        super().__init__(model, config_list)

    def quantize_weight(self, weight, config, **kwargs):
        """
        quantize should overload this method to quantize weight tensors.
        This method is effectively hooked to :meth:`forward` of the model.

        Parameters
        ----------
        weight : Tensor
            weight that needs to be quantized
        config : dict
            the configuration for weight quantization
        """

        # Put your code to generate `new_weight` here
        new_weight = ...
        return new_weight

    def quantize_output(self, output, config, **kwargs):
        """
        quantize should overload this method to quantize output.
        This method is effectively hooked to `:meth:`forward` of the model.

        Parameters
        ----------
        output : Tensor
            output that needs to be quantized
        config : dict
            the configuration for output quantization
        """

        # Put your code to generate `new_output` here
        new_output = ...
        return new_output

    def quantize_input(self, *inputs, config, **kwargs):
        """
        quantize should overload this method to quantize input.
        This method is effectively hooked to :meth:`forward` of the model.

        Parameters
        ----------
        inputs : Tensor
            inputs that needs to be quantized
        config : dict
            the configuration for inputs quantization
        """

        # Put your code to generate `new_input` here
        new_input = ...
        return new_input

    def update_epoch(self, epoch_num):
        pass

    def step(self):
        """
        Can do some processing based on the model or weights binded
        in the func bind_model
        """
        pass

# %%
# Customize backward function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sometimes it's necessary for a quantization operation to have a customized backward function,
# such as `Straight-Through Estimator <https://stackoverflow.com/questions/38361314/the-concept-of-straight-through-estimator-ste>`__\ ,
# user can customize a backward function as follow:

from nni.compression.pytorch.compressor import Quantizer, QuantGrad, QuantType

class ClipGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type):
        """
        This method should be overrided by subclass to provide customized backward function,
        default implementation is Straight-Through Estimator
        Parameters
        ----------
        tensor : Tensor
            input of quantization operation
        grad_output : Tensor
            gradient of the output of quantization operation
        quant_type : QuantType
            the type of quantization, it can be `QuantType.INPUT`, `QuantType.WEIGHT`, `QuantType.OUTPUT`,
            you can define different behavior for different types.
        Returns
        -------
        tensor
            gradient of the input of quantization operation
        """

        # for quant_output function, set grad to zero if the absolute value of tensor is larger than 1
        if quant_type == QuantType.OUTPUT:
            grad_output[tensor.abs() > 1] = 0
        return grad_output

class _YourQuantizer(Quantizer):
    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        # set your customized backward function to overwrite default backward function
        self.quant_grad = ClipGrad

# %%
# If you do not customize ``QuantGrad``, the default backward is Straight-Through Estimator. 
