# Customize New Compression Algorithm

```eval_rst
.. contents::
```

In order to simplify the process of writing new compression algorithms, we have designed simple and flexible programming interface, which covers pruning and quantization. Below, we first demonstrate how to customize a new pruning algorithm and then demonstrate how to customize a new quantization algorithm.

**Important Note** To better understand how to customize new pruning/quantization algorithms, users should first understand the framework that supports various pruning algorithms in NNI. Reference [Framework overview of model compression](https://nni.readthedocs.io/en/latest/Compression/Framework.html)


## Customize a new pruning algorithm

Implementing a new pruning algorithm requires implementing a `weight masker` class which shoud be a subclass of `WeightMasker`, and a `pruner` class, which should be a subclass `Pruner`.

An implementation of `weight masker` may look like this:

```python
class MyMasker(WeightMasker):
    def __init__(self, model, pruner):
        super().__init__(model, pruner)
        # You can do some initialization here, such as collecting some statistics data
        # if it is necessary for your algorithms to calculate the masks.

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        # calculate the masks based on the wrapper.weight, and sparsity, 
        # and anything else
        # mask = ...
        return {'weight_mask': mask}
```

You can reference nni provided [weight masker](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression/torch/pruning/structured_pruning.py) implementations to implement your own weight masker.

A basic `pruner` looks likes this:

```python
class MyPruner(Pruner):
    def __init__(self, model, config_list, optimizer):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        # construct a weight masker instance
        self.masker = MyMasker(model, self)

    def calc_mask(self, wrapper, wrapper_idx=None):
        sparsity = wrapper.config['sparsity']
        if wrapper.if_calculated:
            # Already pruned, do not prune again as a one-shot pruner
            return None
        else:
            # call your masker to actually calcuate the mask for this layer
            masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)
            wrapper.if_calculated = True
            return masks

```

Reference nni provided [pruner](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression/torch/pruning/one_shot.py) implementations to implement your own pruner class.


***

## Customize a new quantization algorithm

To write a new quantization algorithm, you can write a class that inherits `nni.compression.torch.Quantizer`. Then, override the member functions with the logic of your algorithm. The member function to override is `quantize_weight`. `quantize_weight` directly returns the quantized weights rather than mask, because for quantization the quantized weights cannot be obtained by applying mask.

```python
from nni.compression.torch import Quantizer

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

        return new_input

    def update_epoch(self, epoch_num):
        pass

    def step(self):
        """
        Can do some processing based on the model or weights binded
        in the func bind_model
        """
        pass
```

### Customize backward function

Sometimes it's necessary for a quantization operation to have a customized backward function, such as [Straight-Through Estimator](https://stackoverflow.com/questions/38361314/the-concept-of-straight-through-estimator-ste), user can customize a backward function as follow:

```python
from nni.compression.torch.compressor import Quantizer, QuantGrad, QuantType

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
            the type of quantization, it can be `QuantType.QUANT_INPUT`, `QuantType.QUANT_WEIGHT`, `QuantType.QUANT_OUTPUT`,
            you can define different behavior for different types.
        Returns
        -------
        tensor
            gradient of the input of quantization operation
        """

        # for quant_output function, set grad to zero if the absolute value of tensor is larger than 1
        if quant_type == QuantType.QUANT_OUTPUT: 
            grad_output[torch.abs(tensor) > 1] = 0
        return grad_output


class YourQuantizer(Quantizer):
    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        # set your customized backward function to overwrite default backward function
        self.quant_grad = ClipGrad

```

If you do not customize `QuantGrad`, the default backward is Straight-Through Estimator. _Coming Soon_ ...
