# 模型压缩教程

```eval_rst
.. contents::
```

本教程中，[第一部分](#模型压缩快速入门)会简单介绍 NNI 上模型压缩的用法。 然后在[第二部分](#使用指南)中进行详细介绍。

## 模型压缩快速入门

NNI 为模型压缩提供了非常简单的 API。 压缩包括剪枝和量化算法。 算法的用法相同，这里以 [slim Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#slim-pruner) 为例来介绍。

### 编写配置

编写配置来指定要剪枝的层。 以下配置表示剪枝所有的 `BatchNorm2d`，稀疏度设为 0.7，其它层保持不变。

```python
configure_list = [{
    'sparsity': 0.7,
    'op_types': ['BatchNorm2d'],
}]
```

配置说明在[这里](#config-list-说明)。 注意，不同的 Pruner 可能有自定义的配置字段，例如，AGP Pruner 有 `start_epoch`。 详情参考每个 Pruner 的[使用](./Pruner.md)，来调整相应的配置。

### 选择压缩算法

选择 Pruner 来修剪模型。 首先，使用模型来初始化 Pruner，并将配置作为参数传入，然后调用 `compress()` 来压缩模型。

```python
pruner = SlimPruner(model, configure_list)
model = pruner.compress()
```

然后，使用正常的训练方法来训练模型 （如，SGD），剪枝在训练过程中是透明的。 一些 Pruner 只在最开始剪枝一次，接下来的训练可被看作是微调优化。 有些 Pruner 会迭代的对模型剪枝，在训练过程中逐步修改掩码。

### 导出压缩结果

训练完成后，可获得剪枝后模型的精度。 可将模型权重到处到文件，同时将生成的掩码也导出到文件。 也支持导出 ONNX 模型。

```python
pruner.export_model(model_path='pruned_vgg19_cifar10.pth', mask_path='mask_vgg19_cifar10.pth')
```

模型的完整示例代码在[这里](https://github.com/microsoft/nni/blob/master/examples/model_compress/model_prune_torch.py)

### 加速模型

掩码实际上并不能加速模型。 要基于导出的掩码，来对模型加速，因此，NNI 提供了 API 来加速模型。 在模型上调用 `apply_compression_results` 后，模型会变得更小，推理延迟也会减小。

```python
from nni.compression.torch import apply_compression_results
apply_compression_results(model, 'mask_vgg19_cifar10.pth')
```

参考[这里](ModelSpeedup.md)，了解详情。

## 使用指南

将压缩应用到模型的示例代码如下：

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


可使用 `nni.compression` 中的其它压缩算法。 此算法分别在 `nni.compression.torch` 和 `nni.compression.tensorflow` 中实现，支持 PyTorch 和 TensorFlow（部分支持）。 参考 [Pruner](./Pruner.md) 和 [Quantizer](./Quantizer.md) 进一步了解支持的算法。 此外，如果要使用知识蒸馏算法，可参考 [KD 示例](../TrialExample/KDExample.md)

压缩算法首先通过传入 `config_list` 来实例化。 `config_list` 会稍后介绍。

函数调用 `pruner.compress()` 来修改用户定义的模型（在 Tensorflow 中，通过 `tf.get_default_graph()` 来获得模型，而 PyTorch 中 model 是定义的模型类），并修改模型来插入 mask。 然后运行模型时，这些掩码即会生效。 掩码可在运行时通过算法来调整。

*注意，`pruner.compress` 只会在模型权重上直接增加掩码，不包括调优的逻辑。 如果要想调优压缩后的模型，需要在 `pruner.compress` 后增加调优的逻辑。*

### `config_list` 说明

用户可为压缩算法指定配置 (即, `config_list`)。 例如，压缩模型时，用户可能希望指定稀疏率，为不同类型的操作指定不同的稀疏比例，排除某些类型的操作，或仅压缩某类操作。 配置规范可用于表达此类需求。 可将其视为一个 Python 的 `list` 对象，其中每个元素都是一个 `dict` 对象。

`list` 中的 `dict` 会依次被应用，也就是说，如果一个操作出现在两个配置里，后面的 `dict` 会覆盖前面的配置。

`dict` 中有不同的键值。 以下是所有压缩算法都支持的：

* __op_types__：指定要压缩的操作类型。 'default' 表示使用算法的默认设置。
* __op_names__：指定需要压缩的操作的名称。 如果没有设置此字段，操作符不会通过名称筛选。
* __exclude__：默认为 False。 如果此字段为 True，表示要通过类型和名称，将一些操作从压缩中排除。

其它算法的键值，可参考[剪枝算法](./Pruner.md)和[量化算法](./Quantizer.md)，查看每个算法的键值。

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

#### 其它量化算法字段

**如果使用量化算法，则需要设置更多键值。 如果使用剪枝算法，则可以忽略这些键值**

* __quant_types__ : 字符串列表。

要应用量化的类型，当前支持 'weight', 'input', 'output'。 'weight' 是指将量化操作应用到 module 的权重参数上。 'input' 是指对 module 的 forward 方法的输入应用量化操作。 'output' 是指将量化运法应用于模块 forward 方法的输出，有些论文中将其称为 '激活（activation）'。

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

### 更新优化状态的 API

一些压缩算法使用 Epoch 来控制压缩过程（如，[AGP](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#agp-pruner)），一些算法需要在每个批处理步骤后执行一些逻辑。 因此，NNI 提供了两个 API：`pruner.update_epoch(epoch)` 和 `pruner.step()`。

`update_epoch` 会在每个 Epoch 时调用，而 `step` 会在每次批处理后调用。 注意，大多数算法不需要调用这两个 API。 详细情况可参考具体算法文档。 对于不需要这两个 API 的算法，可以调用它们，但不会有实际作用。

### 导出压缩模型

使用下列 API 可轻松将压缩后的模型导出，稀疏模型的 `state_dict` 会保存在 `model.pth` 文件中，可通过 `torch.load('model.pth')` 加载。 在导出的 `model.pth` 中，被掩码遮盖的权重为零。

```
pruner.export_model(model_path='model.pth')
```

`mask_dict` 和 `onnx` 格式的剪枝模型（需要指定 `input_shape`）可这样导出：

```python
pruner.export_model(model_path='model.pth', mask_path='mask.pth', onnx_path='model.onnx', input_shape=[1, 1, 28, 28])
```

如果需要实际加速压缩后的模型，参考 [NNI 模型加速](./ModelSpeedup.md)。