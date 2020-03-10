# 设计文档

## 概述
模型压缩框架有两个主要组件： `Pruner` 和 `module 的包装`。

### Pruner
`Pruner` 用于：
1. 提供 `cal_mask` 方法来计算权重和偏差的掩码（mask）。
2. 根据配置，用 `module 的包装`来替换原始的 module。
3. 修改优化器，来在 `step` 方法被调用时，调用 `cal_mask`。

### module 的包装
`module 的包装` 包含：
1. 原始的 module
2. `cal_mask` 使用的一些缓存
3. 新的 forward 方法，用于在运行原始的 forward 方法前应用掩码。

使用 `module 包装`的原因：
1. 计算掩码所需要的 `cal_mask` 方法需要一些缓存，这些缓存需要注册在 `module 包装`里，这样就不需要修改原始的 module。
2. 新的 `forward` 方法用来在原始 `forward` 调用前，将掩码应用到权重上。

## 工作原理
基本的 Pruner 用法：
```python
configure_list = [{
    'sparsity': 0.7,
    'op_types': ['BatchNorm2d'],
}]

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
pruner = SlimPruner(model, configure_list, optimizer)
model = pruner.compress()
```

Pruner 接收模型，配置和优化器作为参数。 在 `__init__` 方法中，优化器的 `step` 方法会被一个会调用 `cal_mask` 的新的 `step` 方法替换。 同样，所有 module 都会检查它们是否被配置为需要剪枝。如果 module 需要被剪枝，就会用 `module 包装`来替换它。 之后，会返回新的模型和优化器，并进行训练。 `compress` 方法会计算默认的掩码。

## 实现新的剪枝算法
要实现新的剪枝算法，需要继承 `Pruner` 来实现新的类，并重载 `cal_mask` 方法。 `cal_mask` 会被 `optimizer.step` 方法调用。 `Pruner` 基类提供了上述的基本功能，如替换 module 和优化器。

基础的 Pruner 如下所示：
```python
class NewPruner(Pruner):
    def __init__(self, model, config_list, optimizer)
        super().__init__(model, config_list, optimizer)
        # 进行初始化

    def calc_mask(self, wrapper, **kwargs):
        # 计算 weight_mask
        wrapper.weight_mask = weight_mask
```
### 设置包装的属性
有时，`cal_mask` 需要保存一些状态数据，可以像 PyTorch 的 module 一样，使用 `set_wrappers_attribute` API 来注册属性。 这些缓存会注册到 `module 包装`中。 用户可以通过 `module 包装`来直接访问这些缓存。

```python
class NewPruner(Pruner):
    def __init__(self, model, config_list, optimizer):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)

    def calc_mask(self, wrapper):
        # 计算 weight_mask
        if wrapper.if_calculated:
            pass
        else:
            wrapper.if_calculated = True
            # 更新掩码
```

### 在 forward 时收集数据
有时，需要在 forward 方法中收集数据，例如，需要激活的平均值。 这时，可以为 module 增加定制的收集方法。

```python
class ActivationRankFilterPruner(Pruner):
    def __init__(self, model, config_list, optimizer, activation='relu', statistics_batch_num=1):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.set_wrappers_attribute("collected_activation", [])
        self.statistics_batch_num = statistics_batch_num

        def collector(module_, input_, output):
            if len(module_.collected_activation) < self.statistics_batch_num:
                module_.collected_activation.append(self.activation(output.detach().cpu()))
        self.add_activation_collector(collector)
        assert activation in ['relu', 'relu6']
        if activation == 'relu':
            self.activation = torch.nn.functional.relu
        elif activation == 'relu6':
            self.activation = torch.nn.functional.relu6
        else:
            self.activation = None
```
收集函数会在每次 forward 方法运行时调用。

还可这样来移除收集方法：
```python
collector_id = self.add_activation_collector(collector)
# ...
self.remove_activation_collector(collector_id)
```

### 多 GPU 支持
在多 GPU 训练中，缓存和参数会在每次 `forward` 方法被调用时，复制到多个 GPU 上。 如果缓存和参数要在 `forward` 更新，就需要通过`原地`更新来提高效率。 因为 `cal_mask` 会在 `optimizer.step` 方法中的调用，会在 `forward` 方法后才被调用，且只会发生在单 GPU 上，因此它天然的就支持多 GPU 的情况。