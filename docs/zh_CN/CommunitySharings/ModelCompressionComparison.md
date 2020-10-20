# 滤波器剪枝算法比较

为了初步了解各种滤波器剪枝算法的性能，在一些基准模型和数据集上使用各种剪枝算法进行了广泛的实验。 此文档中展示了实验结果。 此外，还对这些实验的复现提供了友好的说明，以促进对这项工作的进一步贡献。

## 实验设置

实验使用以下剪枝器/数据集/模型进行:

* 模型：[VGG16, ResNet18, ResNet50](https://github.com/microsoft/nni/tree/master/examples/model_compress/models/cifar10)

* 数据集：CIFAR-10

* 剪枝器：
    - 剪枝器包括：
        - 迭代式剪枝器 : `SimulatedAnnealing Pruner`, `NetAdapt Pruner`, `AutoCompress Pruner`。 给定总体稀疏度要求，这类剪枝器可以在不同层中自动分配稀疏度。
        - 单轮剪枝器：`L1Filter Pruner`，`L2Filter Pruner`，`FPGM Pruner`。 每层的稀疏度与实验设置的总体稀疏度相同。
    - 这里只比较 **filter pruning** 的剪枝效果。

    对于迭代式剪枝器，使用 `L1Filter Pruner` 作为基础算法。 也就是说, 在迭代式剪枝器决定了稀疏度分布之后，使用 `L1Filter Pruner` 进行真正的剪枝。

    - All the pruners listed above are implemented in [nni](https://github.com/microsoft/nni/tree/master/docs/en_US/Compression/Overview.md).

## 实验结果

对于每一个数据集/模型/剪枝器的组合，设置不同的目标稀疏度对模型进行剪枝。

这里展示了**权重数量 - 性能**曲线，还展示了**FLOPs - 性能**曲线。 同时在图上画出论文 [AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates](http://arxiv.org/abs/1907.03141) 中对 VGG16 和 ResNet18 在 CIFAR-10 上的剪枝结果作为对比。

实验结果如下图所示：

CIFAR-10, VGG16:

![](../../../examples/model_compress/comparison_of_pruners/img/performance_comparison_vgg16.png)

CIFAR-10, ResNet18:

![](../../../examples/model_compress/comparison_of_pruners/img/performance_comparison_resnet18.png)

CIFAR-10, ResNet50:

![](../../../examples/model_compress/comparison_of_pruners/img/performance_comparison_resnet50.png)

## 分析

从实验结果中，得到以下结论：

* 如果稀疏度是通过限制参数量，那么迭代式剪枝器 ( `AutoCompress Pruner` , `SimualatedAnnealing Pruner` ) 比单轮剪枝器表现好。 但是在以 FLOPs 稀疏度为标准的情况下，它们相比于单轮剪枝器就没有了优势，因为当前的这些剪枝算法都是根据参数稀疏度来剪枝的。
* 在上述实验中，简单的单轮剪枝器 `L1Filter Pruner` , `L2Filter Pruner` , `FPGM Pruner` 表现比较相近。
* `NetAdapt Pruner` 无法达到比较高的压缩率。 因为它的机制是一次迭代只剪枝一层。 这就导致如果每次迭代剪掉的稀疏度远小于指定的总的稀疏度的话，会导致不可接受的剪枝复杂度。

## 实验复现

### 实现细节

* 实验结果都是在 NNI 中使用剪枝器的默认配置收集的，这意味着当我们在 NNI 中调用一个剪枝器类时，我们不会更改任何默认的类参数。

* Both FLOPs and the number of parameters are counted with [Model FLOPs/Parameters Counter](https://github.com/microsoft/nni/tree/master/docs/en_US/Compression/CompressionUtils.md#model-flopsparameters-counter) after [model speed up](https://github.com/microsoft/nni/tree/master/docs/en_US/Compression/ModelSpeedup.md). 这避免了依据掩码模型计算的潜在问题。

* 实验代码在[这里](https://github.com/microsoft/nni/tree/master/examples/model_compress/auto_pruners_torch.py)。

### 实验结果展示

* 如果遵循[示例](https://github.com/microsoft/nni/tree/master/examples/model_compress/auto_pruners_torch.py)的做法，对于每一次剪枝实验，实验结果将以JSON格式保存如下：
    ``` json
    {
        "performance": {"original": 0.9298, "pruned": 0.1, "speedup": 0.1, "finetuned": 0.7746}, 
        "params": {"original": 14987722.0, "speedup": 167089.0}, 
        "flops": {"original": 314018314.0, "speedup": 38589922.0}
    }
    ```

* 实验结果保存在[这里](https://github.com/microsoft/nni/tree/master/examples/model_compress/comparison_of_pruners)。 可以参考[分析](https://github.com/microsoft/nni/blob/master/examples/model_compress/comparison_of_pruners/analyze.py)来绘制新的性能比较图。

## 贡献

### 待办事项

* 有 FLOPS/延迟 限制的剪枝器
* 更多剪枝算法/数据集/模型

### 问题
关于算法实现及实验问题，请[发起 issue](https://github.com/microsoft/nni/issues/new/)。
