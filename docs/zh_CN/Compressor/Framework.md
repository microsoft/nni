# 设计文档

## 概述

下列示例展示了如何使用 Pruner：

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

Pruner 接收 `model`, `config_list` 以及 `optimizer` 参数。 通过往 `optimizer.step()` 上增加回调，在训练过程中根据 `config_list` 来对模型剪枝。

从实现上来看，Pruner 由 `权重掩码`实例和若干个 `module 包装`实例组成。

### 权重掩码

`权重掩码`是剪枝算法的实现，可将由 `module 包装`所包装起来的一层根据稀疏度进行修建。

### module 的包装

`module 的包装` 包含：

1. 原始的 module
2. `calc_mask` 使用的一些缓存
3. 新的 forward 方法，用于在运行原始的 forward 方法前应用掩码。

使用 `module 包装`的原因：

1. 计算掩码所需要的 `calc_mask` 方法需要一些缓存，这些缓存需要注册在 `module 包装`里，这样就不需要修改原始的 module。
2. 新的 `forward` 方法用来在原始 `forward` 调用前，将掩码应用到权重上。

### Pruner

`Pruner` 用于：

1. 管理、验证 config_list.
2. 使用 `module 包装`来包装模型层，并在 `optimizer.step` 上添加回调
3. 使用`权重掩码`在剪枝时计算层的掩码。
4. 导出剪枝后模型的权重和掩码。

## 实现新的剪枝算法

要实现新的剪枝算法，需要实现`权重掩码`类，它是 `WeightMasker` 的子类，以及`Pruner` 类，它是 `Pruner` 的子类。

`权重掩码`的实现如下：

```python
class MyMasker(WeightMasker):
    def __init__(self, model, pruner):
        super().__init__(model, pruner)
        # 此处可初始化，如为算法收集计算权重所需要的统计信息。

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        # 根据 wrapper.weight, 和 sparsity, 
        # 及其它信息来计算掩码
        # mask = ...
        return {'weight_mask': mask}
```

参考 NNI 提供的[权重掩码](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression/torch/pruning/structured_pruning.py)来实现自己的。

基本的 Pruner 如下所示：

```python
class MyPruner(Pruner):
    def __init__(self, model, config_list, optimizer):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        # 创建权重掩码实例
        self.masker = MyMasker(model, self)

    def calc_mask(self, wrapper, wrapper_idx=None):
        sparsity = wrapper.config['sparsity']
        if wrapper.if_calculated:
            # 如果是一次性剪枝算法，不需要再次剪枝
            return None
        else:
            # 调用掩码函数来实际计算当前层的掩码
            masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)
            wrapper.if_calculated = True
            return masks

```

参考 NNI 提供的[Pruner](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression/torch/pruning/one_shot.py) 来实现自己的。

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

### 多 GPU 支持

在多 GPU 训练中，缓存和参数会在每次 `forward` 方法被调用时，复制到多个 GPU 上。 如果缓存和参数要在 `forward` 更新，就需要通过`原地`更新来提高效率。 因为 `calc_mask` 会在 `optimizer.step` 方法中的调用，会在 `forward` 方法后才被调用，且只会发生在单 GPU 上，因此它天然的就支持多 GPU 的情况。
