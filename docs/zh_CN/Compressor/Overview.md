# 使用 NNI 进行模型压缩
随着更多层和节点大型神经网络的使用，降低其存储和计算成本变得至关重要，尤其是对于某些实时应用程序。 模型压缩可用于解决此问题。

我们很高兴的宣布，基于 NNI 的模型压缩工具发布了试用版本。该版本仍处于试验阶段，根据用户反馈会进行改进。 诚挚邀请您使用、反馈，或有更多贡献。

NNI 提供了易于使用的工具包来帮助用户设计并使用压缩算法。 当前支持基于 PyTorch 的统一接口。 只需要添加几行代码即可压缩模型。 NNI 中也内置了一些流程的模型压缩算法。 用户还可以通过 NNI 强大的自动调参功能来找到最好的压缩后的模型，详见[自动模型压缩](./AutoCompression.md)。 另外，用户还能使用 NNI 的接口，轻松定制新的压缩算法，详见[教程](#customize-new-compression-algorithms)。

模型压缩方面的综述可参考：[Recent Advances in Efficient Computation of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1802.00939.pdf)。

## 支持的算法

NNI 提供了几种压缩算法，包括剪枝和量化算法：

**剪枝**

剪枝算法通过删除冗余权重或层通道来压缩原始网络，从而降低模型复杂性并解决过拟合问题。

| 名称                                                                           | 算法简介                                                                                                                                    |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [Level Pruner](./Pruner.md#level-pruner)                                     | 根据权重的绝对值，来按比例修剪权重。                                                                                                                      |
| [AGP Pruner](./Pruner.md#agp-pruner)                                         | 自动的逐步剪枝（是否剪枝的判断：基于对模型剪枝的效果）[参考论文](https://arxiv.org/abs/1710.01878)                                                                     |
| [Lottery Ticket Pruner](./Pruner.md#agp-pruner)                              | "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" 提出的剪枝过程。 它会反复修剪模型。 [参考论文](https://arxiv.org/abs/1803.03635)  |
| [FPGM Pruner](./Pruner.md#fpgm-pruner)                                       | Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration [参考论文](https://arxiv.org/pdf/1811.00250.pdf)    |
| [L1Filter Pruner](./Pruner.md#l1filter-pruner)                               | 在卷积层中具有最小 L1 权重规范的剪枝过滤器（用于 Efficient Convnets 的剪枝过滤器） [参考论文](https://arxiv.org/abs/1608.08710)                                          |
| [L2Filter Pruner](./Pruner.md#l2filter-pruner)                               | 在卷积层中具有最小 L2 权重规范的剪枝过滤器                                                                                                                 |
| [ActivationAPoZRankFilterPruner](./Pruner.md#ActivationAPoZRankFilterPruner) | 基于指标 APoZ（平均百分比零）的剪枝过滤器，该指标测量（卷积）图层激活中零的百分比。 [参考论文](https://arxiv.org/abs/1607.03250)                                                   |
| [ActivationMeanRankFilterPruner](./Pruner.md#ActivationMeanRankFilterPruner) | 基于计算输出激活最小平均值指标的剪枝过滤器                                                                                                                   |
| [Slim Pruner](./Pruner.md#slim-pruner)                                       | 通过修剪 BN 层中的缩放因子来修剪卷积层中的通道 (Learning Efficient Convolutional Networks through Network Slimming) [参考论文](https://arxiv.org/abs/1708.06519) |


**量化**

量化算法通过减少表示权重或激活所需的精度位数来压缩原始网络，这可以减少计算和推理时间。

| 名称                                                  | 算法简介                                                                                                                                                                       |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Naive Quantizer](./Quantizer.md#naive-quantizer)   | 默认将权重量化为 8 位                                                                                                                                                               |
| [QAT Quantizer](./Quantizer.md#qat-quantizer)       | 为 Efficient Integer-Arithmetic-Only Inference 量化并训练神经网络。 [参考论文](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) |
| [DoReFa Quantizer](./Quantizer.md#dorefa-quantizer) | DoReFa-Net: 通过低位宽的梯度算法来训练低位宽的卷积神经网络。 [参考论文](https://arxiv.org/abs/1606.06160)                                                                                              |
| [BNN Quantizer](./Quantizer.md#BNN-Quantizer)       | 二进制神经网络：使用权重和激活限制为 +1 或 -1 的深度神经网络。 [参考论文](https://arxiv.org/abs/1602.02830)                                                                                               |

## 内置压缩算法的用法

通过简单的示例来展示如何修改 Trial 代码来使用压缩算法。 比如，需要通过 Level Pruner 来将权重剪枝 80%，首先在代码中训练模型前，添加以下内容（[完整代码](https://github.com/microsoft/nni/tree/master/examples/model_compress)）。

PyTorch 代码

```python
from nni.compression.torch import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()
```

TensorFlow 代码

```python
from nni.compression.tensorflow import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(tf.get_default_graph(), config_list)
pruner.compress()
```


可使用 `nni.compression` 中的其它压缩算法。 此算法分别在 `nni.compression.torch` 和 `nni.compression.tensorflow` 中实现，支持 PyTorch 和 TensorFlow。 参考 [Pruner](./Pruner.md) 和 [Quantizer](./Quantizer.md) 进一步了解支持的算法。 此外，如果要使用知识蒸馏算法，可参考 [KD 示例](../TrialExample/KDExample.md)

函数调用 `pruner.compress()` 来修改用户定义的模型（在 Tensorflow 中，通过 `tf.get_default_graph()` 来获得模型，而 PyTorch 中 model 是定义的模型类），并修改模型来插入 mask。 然后运行模型时，这些 mask 即会生效。 mask 可在运行时通过算法来调整。

实例化压缩算法时，会传入 `config_list`。 配置说明如下。

### 压缩算法中的用户配置
压缩模型时，用户可能希望指定稀疏率，为不同类型的操作指定不同的比例，排除某些类型的操作，或仅压缩某类操作。 配置规范可用于表达此类需求。 可将其视为一个 Python 的 `list` 对象，其中每个元素都是一个 `dict` 对象。

`list` 中的 `dict` 会依次被应用，也就是说，如果一个操作出现在两个配置里，后面的 `dict` 会覆盖前面的配置。

#### 通用键值
在每个 `dict` 中，有一些 NNI 压缩算法支持的键值：

* __op_types__：指定要压缩的操作类型。 'default' 表示使用算法的默认设置。
* __op_names__：指定需要压缩的操作的名称。 如果没有设置此字段，操作符不会通过名称筛选。
* __exclude__：默认为 False。 如果此字段为 True，表示要通过类型和名称，将一些操作从压缩中排除。

#### 量化算法的键值
**If you use quantization algorithms, you need to specify more keys. If you use pruning algorithms, you can safely skip these keys**

* __quant_types__ : list of string.

Type of quantization you want to apply, currently support 'weight', 'input', 'output'. 'weight' means applying quantization operation to the weight parameter of modules. 'input' means applying quantization operation to the input of module forward method. 'output' means applying quantization operation to the output of module forward method, which is often called as 'activation' in some papers.

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

### 其它 API

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

You can easily export the compressed model using the following API if you are pruning your model, `state_dict` of the sparse model weights will be stored in `model.pth`, which can be loaded by `torch.load('model.pth')`

```
pruner.export_model(model_path='model.pth')
```

`mask_dict` and pruned model in `onnx` format(`input_shape` need to be specified) can also be exported like this:

```python
pruner.export_model(model_path='model.pth', mask_path='mask.pth', onnx_path='model.onnx', input_shape=[1, 1, 28, 28])
```

## 定制新的压缩算法

To simplify writing a new compression algorithm, we design programming interfaces which are simple but flexible enough. There are interfaces for pruner and quantizer respectively.

### 剪枝算法

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

For the simplest algorithm, you only need to override `calc_mask`. It receives the to-be-compressed layers one by one along with their compression configuration. You generate the mask for this weight in this function and return. Then NNI applies the mask for you.

Some algorithms generate mask based on training progress, i.e., epoch number. We provide `update_epoch` for the pruner to be aware of the training progress. It should be called at the beginning of each epoch.

Some algorithms may want global information for generating masks, for example, all weights of the model (for statistic information). Your can use `self.bound_model` in the Pruner class for accessing weights. If you also need optimizer's information (for example in Pytorch), you could override `__init__` to receive more arguments such as model's optimizer. Then `step` can process or update the information according to the algorithm. You can refer to [source code of built-in algorithms](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/compressors) for example implementations.

### 量化算法

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

If you do not customize `QuantGrad`, the default backward is Straight-Through Estimator. _Coming Soon_ ...

## **Reference and Feedback**
* To [report a bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md) for this feature in GitHub;
* To [file a feature or improvement request](https://github.com/microsoft/nni/issues/new?template=enhancement.md) for this feature in GitHub;
* To know more about [Feature Engineering with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/FeatureEngineering/Overview.md);
* To know more about [NAS with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/NAS/Overview.md);
* To know more about [Hyperparameter Tuning with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/Tuner/BuiltinTuner.md);
