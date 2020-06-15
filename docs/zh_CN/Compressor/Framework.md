# 设计文档

## 概述

Following example shows how to use a pruner:

```python
from nni.compression.torch import LevelPruner

# load a pretrained model or train a model before using a pruner

configure_list = [{
    'sparsity': 0.7,
    'op_types': ['Conv2d', 'Linear'],
}]

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
pruner = LevelPruner(model, configure_list, optimizer)
model = pruner.compress()

# model is ready for pruning, now start finetune the model,
# the model will be pruned during training automatically
```

A pruner receives `model`, `config_list` and `optimizer` as arguments. It prunes the model per the `config_list` during training loop by adding a hook on `optimizer.step()`.

From implementation perspective, a pruner consists of a `weight masker` instance and multiple `module wrapper` instances.

### Weight masker

A `weight masker` is the implementation of pruning algorithms, it can prune a specified layer wrapped by `module wrapper` with specified sparsity.

### Module wrapper

A `module wrapper` is a module containing:

1. 原始的 module
2. some buffers used by `calc_mask`
3. a new forward method that applies masks before running the original forward method.

the reasons to use `module wrapper`:

1. some buffers are needed by `calc_mask` to calculate masks and these buffers should be registered in `module wrapper` so that the original modules are not contaminated.
2. a new `forward` method is needed to apply masks to weight before calling the real `forward` method.

### Pruner

A `pruner` is responsible for:

1. Manage / verify config_list.
2. Use `module wrapper` to wrap the model layers and add hook on `optimizer.step`
3. Use `weight masker` to calculate masks of layers while pruning.
4. Export pruned model weights and masks.

## Implement a new pruning algorithm

Implementing a new pruning algorithm requires implementing a `weight masker` class which shoud be a subclass of `WeightMasker`, and a `pruner` class, which should a subclass `Pruner`.

An implementation of `weight masker` may look like this:

```python
class MyMasker(WeightMasker):
    def __init__(self, model, pruner):
        super().__init__(model, pruner)
        # You can do some initialization here, such as collecting some statistics data
        # if it is necessary for your algorithms to calculate the masks.

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        # calculate the masks based on the wrapper.weight, and sparsity, 
        # and anything else
        # mask = ...
        return {'weight_mask': mask}
```

You can reference nni provided [weight masker](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression/torch/pruning/structured_pruning.py) implementations to implement your own weight masker.

A basic pruner looks likes this:

```python
class MyPruner(Pruner):
    def __init__(self, model, config_list, optimizer):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        # construct a weight masker instance
        self.masker = MyMasker(model, self)

    def calc_mask(self, wrapper, wrapper_idx=None):
        sparsity = wrapper.config['sparsity']
        if wrapper.if_calculated:
            # Already pruned, do not prune again as a one-shot pruner
            return None
        else:
            # call your masker to actually calcuate the mask for this layer
            masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)
            wrapper.if_calculated = True
            return masks

```

Reference nni provided [pruner](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression/torch/pruning/one_shot.py) implementations to implement your own pruner class.

### Set wrapper attribute

Sometimes `calc_mask` must save some state data, therefore users can use `set_wrappers_attribute` API to register attribute just like how buffers are registered in PyTorch modules. These buffers will be registered to `module wrapper`. Users can access these buffers through `module wrapper`. In above example, we use `set_wrappers_attribute` to set a buffer `if_calculated` which is used as flag indicating if the mask of a layer is already calculated.

### Collect data during forward

Sometimes users want to collect some data during the modules' forward method, for example, the mean value of the activation. This can be done by adding a customized collector to module.

```python
class MyMasker(WeightMasker):
    def __init__(self, model, pruner):
        super().__init__(model, pruner)
        # Set attribute `collected_activation` for all wrappers to store
        # activations for each layer
        self.pruner.set_wrappers_attribute("collected_activation", [])
        self.activation = torch.nn.functional.relu

        def collector(wrapper, input_, output):
            # The collected activation can be accessed via each wrapper's collected_activation
            # attribute
            wrapper.collected_activation.append(self.activation(output.detach().cpu()))

        self.pruner.hook_id = self.pruner.add_activation_collector(collector)
```

The collector function will be called each time the forward method runs.

Users can also remove this collector like this:

```python
# Save the collector identifier
collector_id = self.pruner.add_activation_collector(collector)

# When the collector is not used any more, it can be remove using
# the saved collector identifier
self.pruner.remove_activation_collector(collector_id)
```

### Multi-GPU support

On multi-GPU training, buffers and parameters are copied to multiple GPU every time the `forward` method runs on multiple GPU. If buffers and parameters are updated in the `forward` method, an `in-place` update is needed to ensure the update is effective. Since `calc_mask` is called in the `optimizer.step` method, which happens after the `forward` method and happens only on one GPU, it supports multi-GPU naturally.
