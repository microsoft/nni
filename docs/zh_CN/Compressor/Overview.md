# 使用 NNI 进行模型压缩
随着更多层和节点大型神经网络的使用，降低其存储和计算成本变得至关重要，尤其是对于某些实时应用程序。 模型压缩可用于解决此问题。

我们很高兴的宣布，基于 NNI 的模型压缩工具发布了。该版本仍处于试验阶段，会根据用户反馈进行改进。 诚挚邀请您使用、反馈，或有更多贡献。

NNI 提供了易于使用的工具包来帮助用户设计并使用压缩算法。 当前支持基于 PyTorch 的统一接口。 只需要添加几行代码即可压缩模型。 NNI 中也内置了一些流程的模型压缩算法。 用户还可以通过 NNI 强大的自动调参功能来找到最好的压缩后的模型，详见[自动模型压缩](./AutoCompression.md)。 另外，用户还能使用 NNI 的接口，轻松定制新的压缩算法，详见[教程](#customize-new-compression-algorithms)。 关于模型压缩框架如何工作的详情可参考[这里](./Framework.md)。

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
**如果使用量化算法，则需要设置更多键值。 如果使用剪枝算法，则可以忽略这些键值**

* __quant_types__ : 字符串列表。

要应用量化的类型，当前支持 "权重"，"输入"，"输出"。 "权重"是指将量化操作应用到 module 的权重参数上。 "输入" 是指对 module 的 forward 方法的输入应用量化操作。 "输出"是指将量化运法应用于模块 forward 方法的输出，在某些论文中，这种方法称为"激活"。

* __quant_bits__ : int 或 dict {str : int}

量化的位宽，键是量化类型，值是量化位宽度，例如：
```
{
    quant_bits: {
        'weight': 8,
        'output': 4,
        },
}
```
当值为 int 类型时，所有量化类型使用相同的位宽。 例如：
```
{
    quant_bits: 8, # 权重和输出的位宽都为 8 bits
}
```
#### 为每个压缩算法指定的其他键
`dict` 还有一些其它键值，由特定的压缩算法所使用。 例如， [Level Pruner](./Pruner.md#level-pruner) 需要 `sparsity` 键，用于指定修剪的量。


#### 示例
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
from nni.compression.torch.compressor import Quantizer

class YourQuantizer(Quantizer):
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
            需要被量化的张量
        config : dict
            输入量化的配置
        """

        # 生成 `new_input` 的代码

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
#### 定制 backward 函数
有时，量化操作必须自定义 backward 函数，例如 [Straight-Through Estimator](https://stackoverflow.com/questions/38361314/the-concept-of-straight-through-estimator-ste)，可如下定制 backward 函数：

```python
from nni.compression.torch.compressor import Quantizer, QuantGrad, QuantType

class ClipGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type):
        """
        此方法应被子类重载来提供定制的 backward 函数，
        默认实现是 Straight-Through Estimator
        Parameters
        ----------
        tensor : Tensor
            量化操作的输入
        grad_output : Tensor
            量化操作输出的梯度
        quant_type : QuantType
            量化类型，可为 `QuantType.QUANT_INPUT`, `QuantType.QUANT_WEIGHT`, `QuantType.QUANT_OUTPUT`,
            可为不同的类型定义不同的行为。
        Returns
        -------
        tensor
            量化输入的梯度
        """

        # 对于 quant_output 函数，如果张量的绝对值大于 1，则将梯度设置为 0
        if quant_type == QuantType.QUANT_OUTPUT: 
            grad_output[torch.abs(tensor) > 1] = 0
        return grad_output


class YourQuantizer(Quantizer):
    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        # 定制 backward 函数来重载默认的 backward 函数
        self.quant_grad = ClipGrad

```

如果不定制 `QuantGrad`，默认的 backward 为 Straight-Through Estimator。 _即将推出_...

## 参考和反馈
* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)；
* 了解更多关于 [NNI 中的特征工程](../FeatureEngineering/Overview.md)；
* 了解更多关于 [NNI 中的 NAS](../NAS/Overview.md)；
* 了解更多关于 [NNI 中的超参调优](../Tuner/BuiltinTuner.md)；
