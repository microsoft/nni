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
1. some buffers are needed by `cal_mask` to calculate masks and these buffers should be registered in `module wrapper` so that the original modules are not contaminated.
2. a new `forward` method is needed to apply masks to weight before calling the real `forward` method.

## How it works
A basic pruner usage:
```python
configure_list = [{
    'sparsity': 0.7,
    'op_types': ['BatchNorm2d'],
}]

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
pruner = SlimPruner(model, configure_list, optimizer)
model = pruner.compress()
```

A pruner receive model, config and optimizer as arguments. In the `__init__` method, the `step` method of the optimizer is replaced with a new `step` method that calls `cal_mask`. Also, all modules are checked if they need to be pruned based on config. If a module needs to be pruned, then this module is replaced by a `module wrapper`. Afterward, the new model and new optimizer are returned, which can be trained as before. `compress` method will calculate the default masks.

## Implement a new pruning algorithm
Implementing a new pruning algorithm requires implementing a new `pruner` class, which should subclass `Pruner` and override the `cal_mask` method. The `cal_mask` is called by`optimizer.step` method. The `Pruner` base class provided basic functionality listed above, for example, replacing modules and patching optimizer.

A basic pruner look likes this:
```python
class NewPruner(Pruner):
    def __init__(self, model, config_list, optimizer)
        super().__init__(model, config_list, optimizer)
        # do some initialization

    def calc_mask(self, wrapper, **kwargs):
        # do something to calculate weight_mask
        wrapper.weight_mask = weight_mask
```
### Set wrapper attribute
Sometimes `cal_mask` must save some state data, therefore users can use `set_wrappers_attribute` API to register attribute just like how buffers are registered in PyTorch modules. These buffers will be registered to `module wrapper`. Users can access these buffers through `module wrapper`.

```python
class NewPruner(Pruner):
    def __init__(self, model, config_list, optimizer):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)

    def calc_mask(self, wrapper):
        # do something to calculate weight_mask
        if wrapper.if_calculated:
            pass
        else:
            wrapper.if_calculated = True
            # update masks
```

### Collect data during forward
Sometimes users want to collect some data during the modules' forward method, for example, the mean value of the activation. Therefore user can add a customized collector to module.

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
The collector function will be called each time the forward method runs.

Users can also remove this collector like this:
```python
collector_id = self.add_activation_collector(collector)
# ...
self.remove_activation_collector(collector_id)
```

### Multi-GPU support
On multi-GPU training, buffers and parameters are copied to multiple GPU every time the `forward` method runs on multiple GPU. If buffers and parameters are updated in the `forward` method, an `in-place` update is needed to ensure the update is effective. Since `cal_mask` is called in the `optimizer.step` method, which happens after the `forward` method and happens only on one GPU, it supports multi-GPU naturally.