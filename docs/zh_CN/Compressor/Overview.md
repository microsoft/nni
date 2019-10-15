# Compressor
NNI 提供了易于使用的工具包来帮助用户设计并使用压缩算法。 其使用了统一的接口来支持 TensorFlow 和 PyTorch。 只需要添加几行代码即可压缩模型。 NNI 中也内置了一些流程的模型压缩算法。 用户还可以通过 NNI 强大的自动调参功能来找到最好的压缩后的模型，详见[自动模型压缩](./AutoCompression.md)。 另外，用户还能使用 NNI 的接口，轻松定制新的压缩算法，详见[教程](#customize-new-compression-algorithms)。

## 支持的算法
NNI 提供了两种朴素压缩算法以及四种流行的压缩算法，包括 3 种剪枝算法以及 3 种量化算法：

| 名称                                                   | 算法简介                                                                                                                                                                       |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Level Pruner](./Pruner.md#level-pruner)             | 根据权重的绝对值，来按比例修剪权重。                                                                                                                                                         |
| [AGP Pruner](./Pruner.md#agp-pruner)                 | 自动的逐步剪枝（是否剪枝的判断：基于对模型剪枝的效果）[参考论文](https://arxiv.org/abs/1710.01878)                                                                                                        |
| [Sensitivity Pruner](./Pruner.md#sensitivity-pruner) | 为 Efficient Neural Networks 学习权重和连接。 [参考论文](https://arxiv.org/abs/1506.02626)                                                                                              |
| [Naive Quantizer](./Quantizer.md#naive-quantizer)    | 默认将权重量化为 8 位                                                                                                                                                               |
| [QAT Quantizer](./Quantizer.md#qat-quantizer)        | 为 Efficient Integer-Arithmetic-Only Inference 量化并训练神经网络。 [参考论文](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) |
| [DoReFa Quantizer](./Quantizer.md#dorefa-quantizer)  | DoReFa-Net: 通过低位宽的梯度算法来训练低位宽的卷积神经网络。 [参考论文](https://arxiv.org/abs/1606.06160)                                                                                              |

## 内置压缩算法的用法

通过简单的示例来展示如何修改 Trial 代码来使用压缩算法。 比如，需要通过 Level Pruner 来将权重剪枝 80%，首先在代码中训练模型前，添加以下内容（[完整代码](https://github.com/microsoft/nni/tree/master/examples/model_compress)）。

TensorFlow 代码
```python
from nni.compression.tensorflow import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': 'default' }]
pruner = LevelPruner(config_list)
pruner(tf.get_default_graph())
```

PyTorch 代码
```python
from nni.compression.torch import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': 'default' }]
pruner = LevelPruner(config_list)
pruner(model)
```

可使用 `nni.compression` 中的其它压缩算法。 此算法分别在 `nni.compression.torch` 和 `nni.compression.tensorflow` 中实现，支持 PyTorch 和 TensorFlow。 参考 [Pruner](./Pruner.md) 和 [Quantizer](./Quantizer.md) 进一步了解支持的算法。

函数调用 `pruner(model)` 接收用户定义的模型（在 Tensorflow 中，通过 `tf.get_default_graph()` 来获得模型，而 PyTorch 中 model 是定义的模型类），并修改模型来插入 mask。 然后运行模型时，这些 mask 即会生效。 mask 可在运行时通过算法来调整。

实例化压缩算法时，会传入 `config_list`。 配置说明如下。

### 压缩算法中的用户配置

压缩模型时，用户可能希望指定稀疏率，为不同类型的操作指定不同的比例，排除某些类型的操作，或仅压缩某类操作。 配置规范可用于表达此类需求。 可将其视为一个 Python 的 `list` 对象，其中每个元素都是一个 `dict` 对象。 在每个 `dict` 中，有一些 NNI 压缩算法支持的键值：

* __op_types__: This is to specify what types of operations to be compressed. 'default' means following the algorithm's default setting.
* __op_names__: This is to specify by name what operations to be compressed. If this field is omitted, operations will not be filtered by it.
* __exclude__: Default is False. If this field is True, it means the operations with specified types and names will be excluded from the compression.

There are also other keys in the `dict`, but they are specific for every compression algorithm. For example, some , some.

The `dict`s in the `list` are applied one by one, that is, the configurations in latter `dict` will overwrite the configurations in former ones on the operations that are within the scope of both of them.

A simple example of configuration is shown below:
```python
[
    {
        'sparsity': 0.8,
        'op_types': 'default'
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

__[TODO]__ The last API is for users to export the compressed model. You will get a compressed model when you finish the training using this API. It also exports another file storing the values of masks.

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
    def __init__(self, config_list):
        # suggest you to use the NNI defined spec for config
        super().__init__(config_list)

    def bind_model(self, model):
        # this func can be used to remember the model or its weights
        # in member variables, for getting their values during training
        pass

    def calc_mask(self, weight, config, **kwargs):
        # weight is the target weight tensor
        # config is the selected dict object in config_list for this layer
        # kwargs contains op, op_type, and op_name
        # design your mask and return your mask
        return your_mask

    # note for pytorch version, there is no sess in input arguments
    def update_epoch(self, epoch_num, sess):
        pass

    # note for pytorch version, there is no sess in input arguments
    def step(self, sess):
        # can do some processing based on the model or weights binded
        # in the func bind_model
        pass
```

For the simpliest algorithm, you only need to override `calc_mask`. It receives each layer's weight and selected configuration, as well as op information. You generate the mask for this weight in this function and return. Then NNI applies the mask for you.

Some algorithms generate mask based on training progress, i.e., epoch number. We provide `update_epoch` for the pruner to be aware of the training progress.

Some algorithms may want global information for generating masks, for example, all weights of the model (for statistic information), model optimizer's information. NNI supports this requirement using `bind_model`. `bind_model` receives the complete model, thus, it could record any information (e.g., reference to weights) it cares about. Then `step` can process or update the information according to the algorithm. You can refer to [source code of built-in algorithms](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/compressors) for example implementations.

### Quantization algorithm

The interface for customizing quantization algorithm is similar to that of pruning algorithms. The only difference is that `calc_mask` is replaced with `quantize_weight`. `quantize_weight` directly returns the quantized weights rather than mask, because for quantization the quantized weights cannot be obtained by applying mask.

```python
# This is writing a Quantizer in tensorflow.
# For writing a Quantizer in PyTorch, you can simply replace
# nni.compression.tensorflow.Quantizer with
# nni.compression.torch.Quantizer
class YourPruner(nni.compression.tensorflow.Quantizer):
    def __init__(self, config_list):
        # suggest you to use the NNI defined spec for config
        super().__init__(config_list)

    def bind_model(self, model):
        # this func can be used to remember the model or its weights
        # in member variables, for getting their values during training
        pass

    def quantize_weight(self, weight, config, **kwargs):
        # weight is the target weight tensor
        # config is the selected dict object in config_list for this layer
        # kwargs contains op, op_type, and op_name
        # design your quantizer and return new weight
        return new_weight

    # note for pytorch version, there is no sess in input arguments
    def update_epoch(self, epoch_num, sess):
        pass

    # note for pytorch version, there is no sess in input arguments
    def step(self, sess):
        # can do some processing based on the model or weights binded
        # in the func bind_model
        pass

    # you can also design your method
    def your_method(self, your_input):
        #your code

    def bind_model(self, model):
        #preprocess model
```

__[TODO]__ Will add another member function `quantize_layer_output`, as some quantization algorithms also quantize layers' output.

### Usage of user customized compression algorithm

__[TODO]__ ...
