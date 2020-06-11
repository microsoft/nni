# DARTS

## 介绍

论文 [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) 通过可微分的方式来解决架构搜索中的伸缩性挑战。 此方法基于架构的连续放松的表示，从而允许在架构搜索时能使用梯度下降。

为了实现，作者在小批量中交替优化网络权重和架构权重。 还进一步探讨了使用二阶优化（unroll）来替代一阶，来提高性能的可能性。

NNI 的实现基于[官方实现](https://github.com/quark0/darts)以及一个[第三方实现](https://github.com/khanrc/pt.darts)。 NNI 上的 DARTS 设计为可用于任何搜索空间。 与原始论文一样，为 CIFAR10 实现了 CNN 的搜索空间，来作为 DARTS 的实际示例。

## 重现结果

上述示例旨在重现本文中的结果，我们进行了一阶和二阶优化实验。 由于时间限制，我们仅从第二阶段重新训练了*一次**最佳架构*。 我们的结果目前与论文的结果相当。 稍后会增加更多结果

|             | 论文中           | 重现   |
| ----------- | ------------- | ---- |
| 一阶（CIFAR10） | 3.00 +/- 0.14 | 2.78 |
| 二阶（CIFAR10） | 2.76 +/- 0.09 | 2.89 |

## 示例

### CNN 搜索空间

[示例代码](https://github.com/microsoft/nni/tree/master/examples/nas/darts)

```bash
＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
git clone https://github.com/Microsoft/nni.git

# 搜索最好的架构
cd examples/nas/darts
python3 search.py

# 训练最好的架构
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_49.json
```

## 参考

### PyTorch

```eval_rst
..  autoclass:: nni.nas.pytorch.darts.DartsTrainer
    :members:

..  autoclass:: nni.nas.pytorch.darts.DartsMutator
    :members:
```

## 局限性

* DARTS 不支持 DataParallel，若要支持 DistributedDataParallel，则需要定制。
