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

* __op_types__：指定要压缩的操作类型。 'default' 表示使用算法的默认设置。
* __op_names__：指定需要压缩的操作的名称。 如果没有设置此字段，操作符不会通过名称筛选。
* __exclude__：默认为 False。 如果此字段为 True，表示要通过类型和名称，将一些操作从压缩中排除。

`dict` 还有一些其它键值，由特定的压缩算法所使用。 例如：

`list` 中的 `dict` 会依次被应用，也就是说，如果一个操作出现在两个配置里，后面的 `dict` 会覆盖前面的配置。

配置的简单示例如下：

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

其表示压缩操作的默认稀疏度为 0.8，但`op_name1` 和 `op_name2` 会使用 0.6，且不压缩 `op_name3`。

### 其它 API

一些压缩算法使用 Epoch 来控制压缩进度（如[AGP](./Pruner.md#agp-pruner)），一些算法需要在每个批处理步骤后执行一些逻辑。 因此提供了另外两个 API。 一个是 `update_epoch`，可参考下例使用：

TensorFlow 代码

```python
pruner.update_epoch(epoch, sess)
```

PyTorch 代码

```python
pruner.update_epoch(epoch)
```

另一个是 `step`，可在每个批处理后调用 `pruner.step()`。 注意，并不是所有的算法都需要这两个 API，对于不需要它们的算法，调用它们不会有影响。

__[TODO]__ 最后一个 API 可供用户导出压缩后的模型。 当完成训练后使用此 API，可得到压缩后的模型。 同时也可导出另一个文件用来存储 mask 的数值。

## 定制新的压缩算法

为了简化压缩算法的编写，NNI 设计了简单且灵活的接口。 对于 Pruner 和 Quantizer 分别有相应的接口。

### 剪枝算法

要实现新的剪枝算法，根据使用的框架，添加继承于 `nni.compression.tensorflow.Pruner` 或 `nni.compression.torch.Pruner` 的类。 然后，根据算法逻辑来重写成员函数。

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

对于最简单的算法，只需要重写 `calc_mask` 函数。 它可接收每层的权重，并选择对应的配置和操作的信息。 可在此函数中为此权重生成 mask 并返回。 NNI 会应用此 mask。

一些算法根据训练进度来生成 mask，如 Epoch 数量。 Pruner 可使用 `update_epoch` 来了解训练进度。

一些算法可能需要全局的信息来生成 mask，例如模型的所有权重（用于生成统计信息），模型优化器的信息。 可使用 `bind_model` 来支持此类需求。 `bind_model` 接受完整模型作为参数，因而其记录了所有信息（例如，权重的引用）。 然后 `step` 可以根据算法来处理或更新信息。 可参考[内置算法的源码](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/compressors)作为示例。

### 量化算法

定制量化算法的接口与剪枝算法类似。 唯一的不同是使用 `quantize_weight` 替换了 `calc_mask`。 `quantize_weight` 直接返回量化后的权重，而不是 mask。这是因为对于量化算法，量化后的权重不能通过应用 mask 来获得。

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

__[TODO]__ 添加成员函数 `quantize_layer_output`，用于支持量化层输出的量化算法。

### 使用用户自定义的压缩算法

__[TODO]__ ...
