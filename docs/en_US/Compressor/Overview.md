# Model Compression with NNI
As larger neural networks with more layers and nodes are considered, reducing their storage and computational cost becomes critical, especially for some real-time applications. Model compression can be used to address this problem. 

We are glad to introduce model compression toolkit on top of NNI, it's still in the experiment phase which might evolve based on usage feedback. We'd like to invite you to use, feedback and even contribute.

NNI provides an easy-to-use toolkit to help user design and use compression algorithms. It currently supports PyTorch with unified interface. For users to compress their models, they only need to add several lines in their code. There are some popular model compression algorithms built-in in NNI. Users could further use NNI's auto tuning power to find the best compressed model, which is detailed in [Auto Model Compression](./AutoCompression.md). On the other hand, users could easily customize their new compression algorithms using NNI's interface, refer to the tutorial [here](#customize-new-compression-algorithms). Details about how model compression framework works can be found in [here](./Framework.md).

For a survey of model compression, you can refer to this paper: [Recent Advances in Efficient Computation of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1802.00939.pdf).

## Supported algorithms

We have provided several compression algorithms, including several pruning and quantization algorithms:

**Pruning**

Pruning algorithms compress the original network by removing redundant weights or channels of layers, which can reduce model complexity and address the over-ﬁtting issue.

|Name|Brief Introduction of Algorithm|
|---|---|
| [Level Pruner](./Pruner.md#level-pruner) | Pruning the specified ratio on each weight based on absolute values of weights |
| [AGP Pruner](./Pruner.md#agp-pruner) | Automated gradual pruning (To prune, or not to prune: exploring the efficacy of pruning for model compression) [Reference Paper](https://arxiv.org/abs/1710.01878)|
| [Lottery Ticket Pruner](./Pruner.md#agp-pruner) | The pruning process used by "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". It prunes a model iteratively. [Reference Paper](https://arxiv.org/abs/1803.03635)|
| [FPGM Pruner](./Pruner.md#fpgm-pruner) | Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration [Reference Paper](https://arxiv.org/pdf/1811.00250.pdf)|
| [L1Filter Pruner](./Pruner.md#l1filter-pruner) | Pruning filters with the smallest L1 norm of weights in convolution layers (Pruning Filters for Efficient Convnets) [Reference Paper](https://arxiv.org/abs/1608.08710) |
| [L2Filter Pruner](./Pruner.md#l2filter-pruner) | Pruning filters with the smallest L2 norm of weights in convolution layers |
| [ActivationAPoZRankFilterPruner](./Pruner.md#ActivationAPoZRankFilterPruner) | Pruning filters based on the metric APoZ (average percentage of zeros) which measures the percentage of zeros in activations of (convolutional) layers. [Reference Paper](https://arxiv.org/abs/1607.03250) |
| [ActivationMeanRankFilterPruner](./Pruner.md#ActivationMeanRankFilterPruner) | Pruning filters based on the metric that calculates the smallest mean value of output activations |
| [Slim Pruner](./Pruner.md#slim-pruner) | Pruning channels in convolution layers by pruning scaling factors in BN layers(Learning Efficient Convolutional Networks through Network Slimming) [Reference Paper](https://arxiv.org/abs/1708.06519) |


**Quantization**

Quantization algorithms compress the original network by reducing the number of bits required to represent weights or activations, which can reduce the computations and the inference time.

|Name|Brief Introduction of Algorithm|
|---|---|
| [Naive Quantizer](./Quantizer.md#naive-quantizer) |  Quantize weights to default 8 bits |
| [QAT Quantizer](./Quantizer.md#qat-quantizer) | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. [Reference Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)|
| [DoReFa Quantizer](./Quantizer.md#dorefa-quantizer) | DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. [Reference Paper](https://arxiv.org/abs/1606.06160)|
| [BNN Quantizer](./Quantizer.md#BNN-Quantizer) | Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. [Reference Paper](https://arxiv.org/abs/1602.02830)|

## Usage of built-in compression algorithms

We use a simple example to show how to modify your trial code in order to apply the compression algorithms. Let's say you want to prune all weight to 80% sparsity with Level Pruner, you can add the following three lines into your code before training your model ([here](https://github.com/microsoft/nni/tree/master/examples/model_compress) is complete code).

PyTorch code

```python
from nni.compression.torch import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()
```

Tensorflow code

```python
from nni.compression.tensorflow import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(tf.get_default_graph(), config_list)
pruner.compress()
```


You can use other compression algorithms in the package of `nni.compression`. The algorithms are implemented in both PyTorch and Tensorflow, under `nni.compression.torch` and `nni.compression.tensorflow` respectively. You can refer to [Pruner](./Pruner.md) and [Quantizer](./Quantizer.md) for detail description of supported algorithms. Also if you want to use knowledge distillation, you can refer to [KDExample](../TrialExample/KDExample.md)

The function call `pruner.compress()` modifies user defined model (in Tensorflow the model can be obtained with `tf.get_default_graph()`, while in PyTorch the model is the defined model class), and the model is modified with masks inserted. Then when you run the model, the masks take effect. The masks can be adjusted at runtime by the algorithms.

When instantiate a compression algorithm, there is `config_list` passed in. We describe how to write this config below.

### User configuration for a compression algorithm
When compressing a model, users may want to specify the ratio for sparsity, to specify different ratios for different types of operations, to exclude certain types of operations, or to compress only a certain types of operations. For users to express these kinds of requirements, we define a configuration specification. It can be seen as a python `list` object, where each element is a `dict` object. 

The `dict`s in the `list` are applied one by one, that is, the configurations in latter `dict` will overwrite the configurations in former ones on the operations that are within the scope of both of them. 

#### Common keys
In each `dict`, there are some keys commonly supported by NNI compression:

* __op_types__: This is to specify what types of operations to be compressed. 'default' means following the algorithm's default setting.
* __op_names__: This is to specify by name what operations to be compressed. If this field is omitted, operations will not be filtered by it.
* __exclude__: Default is False. If this field is True, it means the operations with specified types and names will be excluded from the compression.

#### Keys for quantization algorithms
**If you use quantization algorithms, you need to specify more keys. If you use pruning algorithms, you can safely skip these keys**

* __quant_types__ : list of string. 

Type of quantization you want to apply, currently support 'weight', 'input', 'output'. 'weight' means applying quantization operation
to the weight parameter of modules. 'input' means applying quantization operation to the input of module forward method. 'output' means applying quantization operation to the output of module forward method, which is often called as 'activation' in some papers.

* __quant_bits__ : int or dict of {str : int}

bits length of quantization, key is the quantization type, value is the quantization bits length, eg. 
```
{
    quant_bits: {
        'weight': 8,
        'output': 4,
        },
}
```
when the value is int type, all quantization types share same bits length. eg. 
```
{
    quant_bits: 8, # weight or output quantization are all 8 bits
}
```
#### Other keys specified for every compression algorithm
There are also other keys in the `dict`, but they are specific for every compression algorithm. For example, [Level Pruner](./Pruner.md#level-pruner) requires `sparsity` key to specify how much a model should be pruned.


#### example
A simple example of configuration is shown below:

```python
[
    {
        'sparsity': 0.8,
        'op_types': ['default']
    },
    {
        'sparsity': 0.6,
        'op_names': ['op_name1', 'op_name2']
    },
    {
        'exclude': True,
        'op_names': ['op_name3']
    }
]
```

It means following the algorithm's default setting for compressed operations with sparsity 0.8, but for `op_name1` and `op_name2` use sparsity 0.6, and please do not compress `op_name3`.

### Other APIs

Some compression algorithms use epochs to control the progress of compression (e.g. [AGP](./Pruner.md#agp-pruner)), and some algorithms need to do something after every minibatch. Therefore, we provide another two APIs for users to invoke. One is `update_epoch`, you can use it as follows:

Tensorflow code

```python
pruner.update_epoch(epoch, sess)
```

PyTorch code

```python
pruner.update_epoch(epoch)
```

The other is `step`, it can be called with `pruner.step()` after each minibatch. Note that not all algorithms need these two APIs, for those that do not need them, calling them is allowed but has no effect.

You can easily export the compressed model using the following API if you are pruning your model, ```state_dict``` of the sparse model weights will be stored in ```model.pth```, which can be loaded by ```torch.load('model.pth')```

```
pruner.export_model(model_path='model.pth')
```

```mask_dict ``` and pruned model in ```onnx``` format(```input_shape``` need to be specified) can also be exported like this:

```python
pruner.export_model(model_path='model.pth', mask_path='mask.pth', onnx_path='model.onnx', input_shape=[1, 1, 28, 28])
```

## Customize new compression algorithms

To simplify writing a new compression algorithm, we design programming interfaces which are simple but flexible enough. There are interfaces for pruner and quantizer respectively.

### Pruning algorithm

If you want to write a new pruning algorithm, you can write a class that inherits `nni.compression.tensorflow.Pruner` or `nni.compression.torch.Pruner` depending on which framework you use. Then, override the member functions with the logic of your algorithm.

```python
# This is writing a pruner in tensorflow.
# For writing a pruner in PyTorch, you can simply replace
# nni.compression.tensorflow.Pruner with
# nni.compression.torch.Pruner
class YourPruner(nni.compression.tensorflow.Pruner):
    def __init__(self, model, config_list):
        """
        Suggest you to use the NNI defined spec for config
        """
        super().__init__(model, config_list)

    def calc_mask(self, layer, config):
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with ``mul()`` operation on the weight.
        This method is effectively hooked to ``forward()`` method of the model.

        Parameters
        ----------
        layer: LayerInfo
            calculate mask for ``layer``'s weight
        config: dict
            the configuration for generating the mask
        """
        return your_mask

    # note for pytorch version, there is no sess in input arguments
    def update_epoch(self, epoch_num, sess):
        pass

    # note for pytorch version, there is no sess in input arguments
    def step(self, sess):
        """
        Can do some processing based on the model or weights binded
        in the func bind_model
        """
        pass
```

For the simplest algorithm, you only need to override ``calc_mask``. It receives the to-be-compressed layers one by one along with their compression configuration. You generate the mask for this weight in this function and return. Then NNI applies the mask for you.

Some algorithms generate mask based on training progress, i.e., epoch number. We provide `update_epoch` for the pruner to be aware of the training progress. It should be called at the beginning of each epoch.

Some algorithms may want global information for generating masks, for example, all weights of the model (for statistic information). Your can use `self.bound_model` in the Pruner class for accessing weights. If you also need optimizer's information (for example in Pytorch), you could override `__init__` to receive more arguments such as model's optimizer. Then `step` can process or update the information according to the algorithm. You can refer to [source code of built-in algorithms](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/compressors) for example implementations.

### Quantization algorithm

The interface for customizing quantization algorithm is similar to that of pruning algorithms. The only difference is that `calc_mask` is replaced with `quantize_weight`. `quantize_weight` directly returns the quantized weights rather than mask, because for quantization the quantized weights cannot be obtained by applying mask.

```python
from nni.compression.torch.compressor import Quantizer

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
#### Customize backward function
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

If you do not customize `QuantGrad`, the default backward is Straight-Through Estimator. 
_Coming Soon_ ...

## Reference and Feedback
* To [report a bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md) for this feature in GitHub;
* To [file a feature or improvement request](https://github.com/microsoft/nni/issues/new?template=enhancement.md) for this feature in GitHub;
* To know more about [Feature Engineering with NNI](../FeatureEngineering/Overview.md);
* To know more about [NAS with NNI](../NAS/Overview.md);
* To know more about [Hyperparameter Tuning with NNI](../Tuner/BuiltinTuner.md);
