# 模型压缩框架概述

```eval_rst
.. contents::
```

下图展示了模型压缩框架的组件概览。

![](../../img/compressor_framework.jpg)

NNI 模型压缩框架中主要有三个组件/类：`Compressor`, `Pruner` 和 `Quantizer`。 下面会逐个详细介绍：

## Compressor

Compressor 是 Pruner 和 Quantizer 的基类，提供了统一的接口，可用同样的方式使用它们。 例如，使用 Pruner：

```python
from nni.compression.torch import LevelPruner

# 读取预训练的模型，或在使用 Pruner 前进行训练。

configure_list = [{
    'sparsity': 0.7,
    'op_types': ['Conv2d', 'Linear'],
}]

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
pruner = LevelPruner(model, configure_list, optimizer)
model = pruner.compress()

# 剪枝已准备好，开始调优模型，
# 模型会在训练过程中自动剪枝
```

使用 Quantizer：
```python
from nni.compression.torch import DoReFaQuantizer

configure_list = [{
    'quant_types': ['weight'],
    'quant_bits': {
        'weight': 8,
    },
    'op_types':['Conv2d', 'Linear']
}]
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
quantizer = DoReFaQuantizer(model, configure_list, optimizer)
quantizer.compress()

```
查看[示例代码](https://github.com/microsoft/nni/tree/master/examples/model_compress)了解更多信息。

`Compressor` 类提供了一些工具函数：

### 设置包装的属性

有时，`calc_mask` 需要保存一些状态数据，可以像 PyTorch 的 module 一样，使用 `set_wrappers_attribute` API 来注册属性。 这些缓存会注册到 `module 包装`中。 用户可以通过 `module 包装`来直接访问这些缓存。 在上述示例中，使用了 `set_wrappers_attribute` 类设置缓冲 `if_calculated`，它用来标识某层的掩码是否已经计算过了。

### 在 forward 时收集数据

有时，需要在 forward 方法中收集数据，例如，需要激活的平均值。 可通过向 module 中添加定制的 Collector 来做到。

```python
class MyMasker(WeightMasker):
    def __init__(self, model, pruner):
        super().__init__(model, pruner)
        # 为所有包装类设置 `collected_activation` 属性
        # 保存所有层的激活值
        self.pruner.set_wrappers_attribute("collected_activation", [])
        self.activation = torch.nn.functional.relu

        def collector(wrapper, input_, output):
            # 通过每个包装的 collected_activation 属性，来评估收到的激活值
            wrapper.collected_activation.append(self.activation(output.detach().cpu()))

        self.pruner.hook_id = self.pruner.add_activation_collector(collector)
```

收集函数会在每次 forward 方法运行时调用。

还可这样来移除收集方法：

```python
# 保存 Collector 的标识
collector_id = self.pruner.add_activation_collector(collector)

# 当 Collector 不再需要后，可以通过保存的 Collector 标识来删除
self.pruner.remove_activation_collector(collector_id)
```

***

## Pruner

Pruner 接收 `model`, `config_list` 以及 `optimizer` 参数。 通过往 `optimizer.step()` 上增加回调，在训练过程中根据 `config_list` 来对模型剪枝。

Pruner 类是 Compressor 的子类，因此它包含了 Compressor 的所有功能，并添加了剪枝所需要的组件，包括：

### 权重掩码

`权重掩码`是剪枝算法的实现，可将由 `module 包装`所包装起来的一层根据稀疏度进行修建。

### 剪枝模块包装

`剪枝 module 的包装` 包含：

1. 原始的 module
2. `calc_mask` 使用的一些缓存
3. 新的 forward 方法，用于在运行原始的 forward 方法前应用掩码。

使用 `module 包装`的原因：

1. 计算掩码所需要的 `calc_mask` 方法需要一些缓存，这些缓存需要注册在 `module 包装`里，这样就不需要修改原始的 module。
2. 新的 `forward` 方法用来在原始 `forward` 调用前，将掩码应用到权重上。

### 剪枝回调

A pruning hook is installed on a pruner when the pruner is constructed, it is used to call pruner's calc_mask method at `optimizer.step()` is invoked.


***

## Quantizer

Quantizer class is also a subclass of `Compressor`, it is used to compress models by reducing the number of bits required to represent weights or activations, which can reduce the computations and the inference time. It contains:

### Quantization module wrapper

Each module/layer of the model to be quantized is wrapped by a quantization module wrapper, it provides a new `forward` method to quantize the original module's weight, input and output.

### Quantization hook

A quantization hook is installed on a quntizer when it is constructed, it is call at `optimizer.step()`.

### Quantization methods

`Quantizer` class provides following methods for subclass to implement quantization algorithms:

```python
class Quantizer(Compressor):
    """
    Base quantizer for pytorch quantizer
    """
    def quantize_weight(self, weight, wrapper, **kwargs):
        """
        quantize should overload this method to quantize weight.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        weight : Tensor
            weight that needs to be quantized
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module
        """
        raise NotImplementedError('Quantizer must overload quantize_weight()')

    def quantize_output(self, output, wrapper, **kwargs):
        """
        quantize should overload this method to quantize output.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        output : Tensor
            output that needs to be quantized
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module
        """
        raise NotImplementedError('Quantizer must overload quantize_output()')

    def quantize_input(self, *inputs, wrapper, **kwargs):
        """
        quantize should overload this method to quantize input.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        inputs : Tensor
            inputs that needs to be quantized
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module
        """
        raise NotImplementedError('Quantizer must overload quantize_input()')

```

***

## Multi-GPU support

On multi-GPU training, buffers and parameters are copied to multiple GPU every time the `forward` method runs on multiple GPU. If buffers and parameters are updated in the `forward` method, an `in-place` update is needed to ensure the update is effective. Since `calc_mask` is called in the `optimizer.step` method, which happens after the `forward` method and happens only on one GPU, it supports multi-GPU naturally.

