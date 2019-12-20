# Compressor

我们很高兴的宣布，基于 NNI 的模型压缩工具发布了 Alpha 版本。该版本仍处于试验阶段，根据用户反馈会进行改进。 诚挚邀请您使用、反馈，或更多贡献。

NNI 提供了易于使用的工具包来帮助用户设计并使用压缩算法。 其使用了统一的接口来支持 TensorFlow 和 PyTorch。 只需要添加几行代码即可压缩模型。 NNI 中也内置了一些流程的模型压缩算法。 用户还可以通过 NNI 强大的自动调参功能来找到最好的压缩后的模型，详见[自动模型压缩](./AutoCompression.md)。 另外，用户还能使用 NNI 的接口，轻松定制新的压缩算法，详见[教程](#customize-new-compression-algorithms)。

## 支持的算法

NNI 提供了几种压缩算法，包括剪枝和量化算法：

**剪枝**

| 名称                                              | 算法简介                                                                                                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [Level Pruner](./Pruner.md#level-pruner)        | 根据权重的绝对值，来按比例修剪权重。                                                                                                                     |
| [AGP Pruner](./Pruner.md#agp-pruner)            | 自动的逐步剪枝（是否剪枝的判断：基于对模型剪枝的效果）[参考论文](https://arxiv.org/abs/1710.01878)                                                                    |
| [L1Filter Pruner](./Pruner.md#l1filter-pruner)  | 剪除卷积层中最不重要的过滤器 (PRUNING FILTERS FOR EFFICIENT CONVNETS)[参考论文](https://arxiv.org/abs/1608.08710)                                        |
| [Slim Pruner](./Pruner.md#slim-pruner)          | 通过修剪 BN 层中的缩放因子来修剪卷积层中的通道 (Learning Efficient Convolutional Networks through Network Slimming)[参考论文](https://arxiv.org/abs/1708.06519) |
| [Lottery Ticket Pruner](./Pruner.md#agp-pruner) | "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" 提出的剪枝过程。 它会反复修剪模型。 [参考论文](https://arxiv.org/abs/1803.03635) |
| [FPGM Pruner](./Pruner.md#fpgm-pruner)          | Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration [参考论文](https://arxiv.org/pdf/1811.00250.pdf)   |

**量化**

| 名称                                                  | 算法简介                                                                                                                                                                       |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Naive Quantizer](./Quantizer.md#naive-quantizer)   | 默认将权重量化为 8 位                                                                                                                                                               |
| [QAT Quantizer](./Quantizer.md#qat-quantizer)       | 为 Efficient Integer-Arithmetic-Only Inference 量化并训练神经网络。 [参考论文](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) |
| [DoReFa Quantizer](./Quantizer.md#dorefa-quantizer) | DoReFa-Net: 通过低位宽的梯度算法来训练低位宽的卷积神经网络。 [参考论文](https://arxiv.org/abs/1606.06160)                                                                                              |

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

使用下列 API 可轻松将压缩后的模型导出，稀疏模型的 `state_dict` 会保存在 `model.pth` 文件中，可通过 `torch.load('model.pth')` 加载。

```
pruner.export_model(model_path='model.pth')
```

`mask_dict` 和 `onnx` 格式的剪枝模型（需要指定 `input_shape`）可这样导出：

```python
pruner.export_model(model_path='model.pth', mask_path='mask.pth', onnx_path='model.onnx', input_shape=[1, 1, 28, 28])
```

## 定制新的压缩算法

为了简化压缩算法的编写，NNI 设计了简单且灵活的接口。 对于 Pruner 和 Quantizer 分别有相应的接口。

### 剪枝算法

要实现新的剪枝算法，根据使用的框架，添加继承于 `nni.compression.tensorflow.Pruner` 或 `nni.compression.torch.Pruner` 的类。 然后，根据算法逻辑来重写成员函数。

```python
# TensorFlow 中定制 Pruner。
# PyTorch 的 Pruner，只需将
# nni.compression.tensorflow.Pruner 替换为
# nni.compression.torch.Pruner
class YourPruner(nni.compression.tensorflow.Pruner):
    def __init__(self, model, config_list):
        """
        建议使用 NNI 定义的规范来配置
        """
        super().__init__(model, config_list)

    def calc_mask(self, layer, config):
        """
        Pruner 需要重载此方法来为权重提供掩码
        掩码必须与权重有相同的形状和类型。
        将对权重执行 ``mul()`` 操作。
        此方法会挂载到模型的 ``forward()`` 方法上。

        Parameters
        ----------
        layer: LayerInfo
            为 ``layer`` 的权重计算掩码
        config: dict
            生成权重所需要的掩码
        """
        return your_mask

    #  PyTorch 版本不需要 sess 参数
    def update_epoch(self, epoch_num, sess):
        pass

    #  PyTorch 版本不需要 sess 参数
    def step(self, sess):
        """
        根据需要可基于 bind_model 方法中的模型或权重进行操作
        """
        pass
```

对于最简单的算法，只需要重写 `calc_mask` 函数。 它会接收需要压缩的层以及其压缩配置。 可在此函数中为此权重生成 mask 并返回。 NNI 会应用此 mask。

一些算法根据训练进度来生成 mask，如 Epoch 数量。 Pruner 可使用 `update_epoch` 来了解训练进度。 应在每个 Epoch 之前调用它。

一些算法可能需要全局的信息来生成 mask，例如模型的所有权重（用于生成统计信息）. 可在 Pruner 类中通过 `self.bound_model` 来访问权重。 如果需要优化器的信息（如在 Pytorch 中），可重载 `__init__` 来接收优化器等参数。 然后 `step` 可以根据算法来处理或更新信息。 可参考[内置算法的源码](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/compressors)作为示例。

### 量化算法

定制量化算法的接口与剪枝算法类似。 唯一的不同是使用 `quantize_weight` 替换了 `calc_mask`。 `quantize_weight` 直接返回量化后的权重，而不是 mask。这是因为对于量化算法，量化后的权重不能通过应用 mask 来获得。

```python
# TensorFlow 中定制 Quantizer。
# PyTorch 的 Quantizer，只需将
# nni.compression.tensorflow.Quantizer 替换为
# nni.compression.torch.Quantizer
class YourQuantizer(nni.compression.tensorflow.Quantizer):
    def __init__(self, model, config_list):
        """
        建议使用 NNI 定义的规范来配置
        """
        super().__init__(model, config_list)

    def quantize_weight(self, weight, config, **kwargs):
        """
        quantize 需要重载此方法来为权重提供掩码
        此方法挂载于模型的 :meth:`forward`。

        Parameters
        ----------
        weight : Tensor
            要被量化的权重
        config : dict
            权重量化的配置
        """

        # 此处逻辑生成 `new_weight`

        return new_weight

    def quantize_output(self, output, config, **kwargs):
        """
        重载此方法输出量化
        此方法挂载于模型的 `:meth:`forward`。

        Parameters
        ----------
        output : Tensor
            需要被量化的输出
        config : dict
            输出量化的配置
        """

        # 实现生成 `new_output`

        return new_output

    def quantize_input(self, *inputs, config, **kwargs):
        """
        重载此方法量化输入
        此方法挂载于模型的 :meth:`forward`。

        Parameters
        ----------
        inputs : Tensor
            需要量化的输入
        config : dict
            输入量化用的配置
        """

        # 实现生成 `new_input`

        return new_input

    # Pytorch 版本不需要 sess 参数
    def update_epoch(self, epoch_num, sess):
        pass

    # Pytorch 版本不需要 sess 参数
    def step(self, sess):
        """
       根据需要可基于 bind_model 方法中的模型或权重进行操作
        """
        pass
```

### 使用用户自定义的压缩算法

__[TODO]__ ...
