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


You can use other compression algorithms in the package of `nni.compression`. The algorithms are implemented in both PyTorch and TensorFlow (partial support on TensorFlow), under `nni.compression.torch` and `nni.compression.tensorflow` respectively. You can refer to [Pruner](./Pruner.md) and [Quantizer](./Quantizer.md) for detail description of supported algorithms. Also if you want to use knowledge distillation, you can refer to [KDExample](../TrialExample/KDExample.md)

A compression algorithm is first instantiated with a `config_list` passed in. The specification of this `config_list` will be described later.

The function call `pruner.compress()` modifies user defined model (in Tensorflow the model can be obtained with `tf.get_default_graph()`, while in PyTorch the model is the defined model class), and the model is modified with masks inserted. Then when you run the model, the masks take effect. The masks can be adjusted at runtime by the algorithms.

*Note that, `pruner.compress` simply adds masks on model weights, it does not include fine tuning logic. If users want to fine tune the compressed model, they need to write the fine tune logic by themselves after `pruner.compress`.*

### `config_list` 说明

Users can specify the configuration (i.e., `config_list`) for a compression algorithm. For example,when compressing a model, users may want to specify the sparsity ratio, to specify different ratios for different types of operations, to exclude certain types of operations, or to compress only a certain types of operations. For users to express these kinds of requirements, we define a configuration specification. It can be seen as a python `list` object, where each element is a `dict` object.

The `dict`s in the `list` are applied one by one, that is, the configurations in latter `dict` will overwrite the configurations in former ones on the operations that are within the scope of both of them.

There are different keys in a `dict`. Some of them are common keys supported by all the compression algorithms:

* __op_types__: This is to specify what types of operations to be compressed. 'default' means following the algorithm's default setting.
* __op_names__: This is to specify by name what operations to be compressed. If this field is omitted, operations will not be filtered by it.
* __exclude__: Default is False. If this field is True, it means the operations with specified types and names will be excluded from the compression.

Some other keys are often specific to a certain algorithms, users can refer to [pruning algorithms](./Pruner.md) and [quantization algorithms](./Quantizer.md) for the keys allowed by each algorithm.

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

It means following the algorithm's default setting for compressed operations with sparsity 0.8, but for `op_name1` and `op_name2` use sparsity 0.6, and do not compress `op_name3`.

#### 其它量化算法字段

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

### APIs for Updating Fine Tuning Status

Some compression algorithms use epochs to control the progress of compression (e.g. [AGP](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#agp-pruner)), and some algorithms need to do something after every minibatch. Therefore, we provide another two APIs for users to invoke: `pruner.update_epoch(epoch)` and `pruner.step()`.

`update_epoch` should be invoked in every epoch, while `step` should be invoked after each minibatch. Note that most algorithms do not require calling the two APIs. Please refer to each algorithm's document for details. For the algorithms that do not need them, calling them is allowed but has no effect.

### 导出压缩模型

You can easily export the compressed model using the following API if you are pruning your model, `state_dict` of the sparse model weights will be stored in `model.pth`, which can be loaded by `torch.load('model.pth')`. In this exported `model.pth`, the masked weights are zero.

```
pruner.export_model(model_path='model.pth')
```

`mask_dict` and pruned model in `onnx` format(`input_shape` need to be specified) can also be exported like this:

```python
pruner.export_model(model_path='model.pth', mask_path='mask.pth', onnx_path='model.onnx', input_shape=[1, 1, 28, 28])
```

If you want to really speed up the compressed model, please refer to [NNI model speedup](./ModelSpeedup.md) for details.