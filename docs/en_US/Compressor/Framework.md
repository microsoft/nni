## Overview
The model compression framework has two main components: `pruner` and `module wrapper`.

### pruner
A `pruner` is responsible for :
1. provide a `cal_mask` method that calculates masks for weight and bias.
2. replace the module with `module wrapper` based on config.
3. modify the optimizer so that the `cal_mask` method is called every time the `step` method is called.

### module wrapper
A `module wrapper` is a module containing :
1. the origin module
2. some buffers used by `cal_mask`
3. a new forward method that applies masks before running the original forward method.

the reasons to use `module wrapper` :
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
Implementing a new pruning algorithm requires implementing a new `pruner` class, which should subclass `Pruner` and override the `cal_mask` method. The `cal_mask` is called by`optimizer.step` method.
The `Pruner` base class provided basic functionality listed above, for example, replacing modules and patching optimizer.

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
### register buffers
Sometimes `cal_mask` must save some state data, therefore users can use `register_buffer` API to register buffers just like how buffers are registered in PyTorch modules. These buffers will be registered to `module wrapper`. Users can access these buffers through `module wrapper`.

```python
class NewPruner(Pruner):
    def __init__(self, model, config_list, optimizer):
        super().__init__(model, config_list, optimizer)
        self.register_buffer("if_calculated", False)
    
    def calc_mask(self, wrapper):
        # do something to calculate weight_mask
        if wrapper.if_calculated:
            pass
        else:
            wrapper.if_calculated = True
            # update masks
```

### Multi-GPU support
On multi-GPU training, buffers and parameters are copied to multiple GPU every time the `forward` method runs on multiple GPU. If buffers and parameters are updated in the `forward` method, an `in-place` update is needed to ensure the update is effective.
Since `cal_mask` is called in the `optimizer.step` method, which happens after the `forward` method and happens only on one GPU, it supports multi-GPU naturally.