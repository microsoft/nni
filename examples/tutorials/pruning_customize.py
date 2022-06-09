"""
Customize Basic Pruner
======================

Users can easily customize a basic pruner in NNI. A large number of basic modules have been provided and can be reused.
Follow the NNI pruning interface, users only need to focus on their creative parts without worrying about other regular modules.

In this tutorial, we show how to customize a basic pruner.

Concepts
--------

NNI abstracts the basic pruning process into three steps, collecting data, calculating metrics, allocating sparsity.
Most pruning algorithms rely on a metric to decide where should be pruned. Using L1 norm pruner as an example,
the first step is collecting model weights, the second step is calculating L1 norm for weight per output channel,
the third step is ranking L1 norm metric and masking the output channels that have small L1 norm.

In NNI basic pruner, these three step is implement as ``DataCollector``, ``MetricsCalculator`` and ``SparsityAllocator``.

-   ``DataCollector``: This module take pruner as initialize parameter.
    It will get the relevant information of the model from the pruner,
    and sometimes it will also hook the model to get input, output or gradient of a layer or a tensor.
    It can also patch optimizer if some special steps need to be executed before or after ``optimizer.step()``.

-   ``MetricsCalculator``: This module will take the data collected from the ``DataCollector``,
    then calculate the metrics. The metric shape is usually reduced from the data shape.
    The ``dim`` taken by ``MetricsCalculator`` means which dimension will be kept after calculate metrics.
    i.e., the collected data shape is (10, 20, 30), and the ``dim`` is 1, then the dimension-1 will be kept,
    the output metrics shape should be (20,).

-   ``SparsityAllocator``: This module take the metrics and generate the masks.
    Different ``SparsityAllocator`` has different masks generation strategies.
    A common and simple strategy is sorting the metrics' values and calculating a threshold according to the configured sparsity,
    mask the positions which metric value smaller than the threshold.
    The ``dim`` taken by ``SparsityAllocator`` means the metrics are for which dimension, the mask will be expanded to weight shape.
    i.e., the metric shape is (20,), the corresponding layer weight shape is (20, 40), and the ``dim`` is 0.
    ``SparsityAllocator`` will first generate a mask with shape (20,), then expand this mask to shape (20, 40).

Simple Example: Customize a Block-L1NormPruner
----------------------------------------------

NNI already have L1NormPruner, but for the reason of reproducing the paper and reducing user configuration items,
it only support pruning layer output channels. In this example, we will customize a pruner that supports block granularity for Linear.

Note that you don't need to implement all these three kinds of tools for each time,
NNI supports many predefined tools, and you can directly use these to customize your own pruner.
This is a tutorial so we show how to define all these three kinds of pruning tools.

Customize the pruning tools used by the pruner at first.
"""

import torch
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import BasicPruner
from nni.algorithms.compression.v2.pytorch.pruning.tools import (
    DataCollector,
    MetricsCalculator,
    SparsityAllocator
)


# This data collector collects weight in wrapped module as data.
# The wrapped module is the module configured in pruner's config_list.
# This implementation is similar as nni.algorithms.compression.v2.pytorch.pruning.tools.WeightDataCollector
class WeightDataCollector(DataCollector):
    def collect(self):
        data = {}
        # get_modules_wrapper will get all the wrapper in the compressor (pruner),
        # it returns a dict with format {wrapper_name: wrapper},
        # use wrapper.module to get the wrapped module.
        for _, wrapper in self.compressor.get_modules_wrapper().items():
            data[wrapper.name] = wrapper.module.weight.data
        # return {wrapper_name: weight_data}
        return data


class BlockNormMetricsCalculator(MetricsCalculator):
    def __init__(self, block_sparse_size):
        # Because we will keep all dimension with block granularity, so fix ``dim=None``,
        # means all dimensions will be kept.
        super().__init__(dim=None, block_sparse_size=block_sparse_size)

    def calculate_metrics(self, data):
        data_length = len(self.block_sparse_size)
        reduce_unfold_dims = list(range(data_length, 2 * data_length))

        metrics = {}
        for name, t in data.items():
            # Unfold t as block size, and calculate L1 Norm for each block.
            for dim, size in enumerate(self.block_sparse_size):
                t = t.unfold(dim, size, size)
            metrics[name] = t.norm(dim=reduce_unfold_dims, p=1)
        # return {wrapper_name: block_metric}
        return metrics


# This implementation is similar as nni.algorithms.compression.v2.pytorch.pruning.tools.NormalSparsityAllocator
class BlockSparsityAllocator(SparsityAllocator):
    def __init__(self, pruner, block_sparse_size):
        super().__init__(pruner, dim=None, block_sparse_size=block_sparse_size, continuous_mask=True)

    def generate_sparsity(self, metrics):
        masks = {}
        for name, wrapper in self.pruner.get_modules_wrapper().items():
            # wrapper.config['total_sparsity'] can get the configured sparsity ratio for this wrapped module
            sparsity_rate = wrapper.config['total_sparsity']
            # get metric for this wrapped module
            metric = metrics[name]
            # mask the metric with old mask, if the masked position need never recover,
            # just keep this is ok if you are new in NNI pruning
            if self.continuous_mask:
                metric *= self._compress_mask(wrapper.weight_mask)
            # convert sparsity ratio to prune number
            prune_num = int(sparsity_rate * metric.numel())
            # calculate the metric threshold
            threshold = torch.topk(metric.view(-1), prune_num, largest=False)[0].max()
            # generate mask, keep the metric positions that metric values greater than the threshold
            mask = torch.gt(metric, threshold).type_as(metric)
            # expand the mask to weight size, if the block is masked, this block will be filled with zeros,
            # otherwise filled with ones
            masks[name] = self._expand_mask(name, mask)
            # merge the new mask with old mask, if the masked position need never recover,
            # just keep this is ok if you are new in NNI pruning
            if self.continuous_mask:
                masks[name]['weight'] *= wrapper.weight_mask
        return masks


# %%
# Customize the pruner.

class BlockL1NormPruner(BasicPruner):
    def __init__(self, model, config_list, block_sparse_size):
        self.block_sparse_size = block_sparse_size
        super().__init__(model, config_list)

    # Implement reset_tools is enough for this pruner.
    def reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = BlockNormMetricsCalculator(self.block_sparse_size)
        if self.sparsity_allocator is None:
            self.sparsity_allocator = BlockSparsityAllocator(self, self.block_sparse_size)


# %%
# Try this pruner.

# Define a simple model.
class TestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))

model = TestModel()
config_list = [{'op_types': ['Linear'], 'total_sparsity': 0.5}]
# use 2x2 block
_, masks = BlockL1NormPruner(model, config_list, [2, 2]).compress()

# show the generated masks
print('fc1 masks:\n', masks['fc1']['weight'])
print('fc2 masks:\n', masks['fc2']['weight'])


# %%
# This time we successfully define a new pruner with pruning block granularity!
# Note that we don't put validation logic in this example, like ``_validate_config_before_canonical``,
# but for a robust implementation, we suggest you involve the validation logic.
