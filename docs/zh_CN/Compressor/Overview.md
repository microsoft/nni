# Compressor

我们很高兴的宣布，基于 NNI 的模型压缩工具发布了 Alpha 版本。该版本仍处于试验阶段，根据用户反馈会进行改进。 诚挚邀请您使用、反馈，或更多贡献。

NNI 提供了易于使用的工具包来帮助用户设计并使用压缩算法。 其使用了统一的接口来支持 TensorFlow 和 PyTorch。 只需要添加几行代码即可压缩模型。 NNI 中也内置了一些流程的模型压缩算法。 用户还可以通过 NNI 强大的自动调参功能来找到最好的压缩后的模型，详见[自动模型压缩](./AutoCompression.md)。 另外，用户还能使用 NNI 的接口，轻松定制新的压缩算法，详见[教程](#customize-new-compression-algorithms)。

## 支持的算法
NNI 提供了两种朴素压缩算法以及三种流行的压缩算法，包括两种剪枝算法以及三种量化算法：

| 名称                                                  | 算法简介                                                                                                                                                                       |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Level Pruner](./Pruner.md#level-pruner)            | 根据权重的绝对值，来按比例修剪权重。                                                                                                                                                         |
| [AGP Pruner](./Pruner.md#agp-pruner)                | 自动的逐步剪枝（是否剪枝的判断：基于对模型剪枝的效果）[参考论文](https://arxiv.org/abs/1710.01878)                                                                                                        |
| [Naive Quantizer](./Quantizer.md#naive-quantizer)   | 默认将权重量化为 8 位                                                                                                                                                               |
| [QAT Quantizer](./Quantizer.md#qat-quantizer)       | 为 Efficient Integer-Arithmetic-Only Inference 量化并训练神经网络。 [参考论文](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) |
| [DoReFa Quantizer](./Quantizer.md#dorefa-quantizer) | DoReFa-Net: 通过低位宽的梯度算法来训练低位宽的卷积神经网络。 [参考论文](https://arxiv.org/abs/1606.06160)                                                                                              |

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

You can use other compression algorithms in the package of `nni.compression`. The algorithms are implemented in both PyTorch and Tensorflow, under `nni.compression.torch` and `nni.compression.tensorflow` respectively. You can refer to [Pruner](./Pruner.md) and [Quantizer](./Quantizer.md) for detail description of supported algorithms.

The function call `pruner(model)` receives user defined model (in Tensorflow the model can be obtained with `tf.get_default_graph()`, while in PyTorch the model is the defined model class), and the model is modified with masks inserted. Then when you run the model, the masks take effect. The masks can be adjusted at runtime by the algorithms.

When instantiate a compression algorithm, there is `config_list` passed in. We describe how to write this config below.

### 压缩算法中的用户配置

When compressing a model, users may want to specify the ratio for sparsity, to specify different ratios for different types of operations, to exclude certain types of operations, or to compress only a certain types of operations. For users to express these kinds of requirements, we define a configuration specification. It can be seen as a python `list` object, where each element is a `dict` object. In each `dict`, there are some keys commonly supported by NNI compression:

* __op_types__：指定要压缩的操作类型。 'default' 表示使用算法的默认设置。
* __op_names__：指定需要压缩的操作的名称。 如果没有设置此字段，操作符不会通过名称筛选。
* __exclude__：默认为 False。 如果此字段为 True，表示要通过类型和名称，将一些操作从压缩中排除。

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

__[TODO]__ The last API is for users to export the compressed model. You will get a compressed model when you finish the training using this API. It also exports another file storing the values of masks.

## 定制新的压缩算法

To simplify writing a new compression algorithm, we design programming interfaces which are simple but flexible enough. There are interfaces for pruner and quantizer respectively.

### 剪枝算法

If you want to write a new pruning algorithm, you can write a class that inherits `nni.compression.tensorflow.Pruner` or `nni.compression.torch.Pruner` depending on which framework you use. Then, override the member functions with the logic of your algorithm.

```python
# TensorFlow 中定制 Pruner。
# 如果要在 PyTorch 中定制 Pruner，可将
# nni.compression.tensorflow.Pruner 替换为
# nni.compression.torch.Pruner
class YourPruner(nni.compression.tensorflow.Pruner):
    def __init__(self, config_list):
        # 建议使用 NNI 定义的规范来进行配置
        super().__init__(config_list)

    def bind_model(self, model):
        # 此函数可通过成员变量，来保存模型和其权重，
        # 从而能在训练过程中获取这些信息。
        pass

    def calc_mask(self, weight, config, **kwargs):
        # weight 是目标的权重张量
        # config 是在 config_list 中为此层选定的 dict 对象
        # kwargs 包括 op, op_type, 和 op_name
        # 实现定制的 mask 并返回
        return your_mask

    # 注意， PyTorch 不需要 sess 参数
    def update_epoch(self, epoch_num, sess):
        pass

    # 注意， PyTorch 不需要 sess 参数
    def step(self, sess):
        # 根据在 bind_model 函数中引用的模型或权重进行一些处理
        pass
```

For the simpliest algorithm, you only need to override `calc_mask`. It receives each layer's weight and selected configuration, as well as op information. You generate the mask for this weight in this function and return. Then NNI applies the mask for you.

Some algorithms generate mask based on training progress, i.e., epoch number. We provide `update_epoch` for the pruner to be aware of the training progress.

Some algorithms may want global information for generating masks, for example, all weights of the model (for statistic information), model optimizer's information. NNI supports this requirement using `bind_model`. `bind_model` receives the complete model, thus, it could record any information (e.g., reference to weights) it cares about. Then `step` can process or update the information according to the algorithm. You can refer to [source code of built-in algorithms](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/compressors) for example implementations.

### 量化算法

The interface for customizing quantization algorithm is similar to that of pruning algorithms. The only difference is that `calc_mask` is replaced with `quantize_weight`. `quantize_weight` directly returns the quantized weights rather than mask, because for quantization the quantized weights cannot be obtained by applying mask.

```python
# TensorFlow 中定制 Quantizer。
# 如果要在 PyTorch 中定制 Quantizer，可将
# nni.compression.tensorflow.Quantizer 替换为
# nni.compression.torch.Quantizer
class YourQuantizer(nni.compression.tensorflow.Quantizer):
    def __init__(self, config_list):
        # 建议使用 NNI 定义的规范来进行配置
        super().__init__(config_list)

    def bind_model(self, model):
        # 此函数可通过成员变量，来保存模型和其权重，
        # 从而能在训练过程中获取这些信息。
        pass

    def quantize_weight(self, weight, config, **kwargs):
        # weight 是目标的权重张量
        # config 是在 config_list 中为此层选定的 dict 对象
        # kwargs 包括 op, op_type, 和 op_name
        # 实现定制的 Quantizer 并返回新的权重
        return new_weight

    # 注意， PyTorch 不需要 sess 参数
    def update_epoch(self, epoch_num, sess):
        pass

    # 注意， PyTorch 不需要 sess 参数
    def step(self, sess):
        # 根据在 bind_model 函数中引用的模型或权重进行一些处理
        pass
```

__[TODO]__ Will add another member function `quantize_layer_output`, as some quantization algorithms also quantize layers' output.

### 使用用户自定义的压缩算法

__[TODO]__ ...
