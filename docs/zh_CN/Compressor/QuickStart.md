# 模型压缩快速入门

NNI 为模型压缩提供了非常简单的 API。 压缩包括剪枝和量化算法。 它们的用法相同，这里通过 slim Pruner 来演示如何使用。

## 编写配置

编写配置来指定要剪枝的层。 以下配置表示剪枝所有的 `BatchNorm2d`，稀疏度设为 0.7，其它层保持不变。

```python
configure_list = [{
    'sparsity': 0.7,
    'op_types': ['BatchNorm2d'],
}]
```

配置说明在[这里](Overview.md#user-configuration-for-a-compression-algorithm)。 注意，不同的 Pruner 可能有自定义的配置字段，例如，AGP Pruner 有 `start_epoch`。 详情参考每个 Pruner 的 [使用](Overview.md#supported-algorithms)，来调整相应的配置。

## 选择压缩算法

选择 Pruner 来修剪模型。 首先，使用模型来初始化 Pruner，并将配置作为参数传入，然后调用 `compress()` 来压缩模型。

```python
pruner = SlimPruner(model, configure_list)
model = pruner.compress()
```

然后，使用正常的训练方法来训练模型 （如，SGD），剪枝在训练过程中是透明的。 一些 Pruner 只在最开始剪枝一次，接下来的训练可被看作是微调优化。 有些 Pruner 会迭代的对模型剪枝，在训练过程中逐步修改掩码。

## 导出压缩结果

训练完成后，可获得剪枝后模型的精度。 可将模型权重到处到文件，同时将生成的掩码也导出到文件。 也支持导出 ONNX 模型。

```python
pruner.export_model(model_path='pruned_vgg19_cifar10.pth', mask_path='mask_vgg19_cifar10.pth')
```

模型的完整示例代码在[这里](https://github.com/microsoft/nni/blob/master/examples/model_compress/model_prune_torch.py)

## 加速模型

掩码实际上并不能加速模型。 要基于导出的掩码，来对模型加速，因此，NNI 提供了 API 来加速模型。 在模型上调用 `apply_compression_results` 后，模型会变得更小，推理延迟也会减小。

```python
from nni.compression.torch import apply_compression_results
apply_compression_results(model, 'mask_vgg19_cifar10.pth')
```

参考[这里](ModelSpeedup.md)，了解详情。