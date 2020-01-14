# DARTS

## 介绍

论文 [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) 通过可微分的方式来解决架构搜索中的伸缩性挑战。 Their method is based on the continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent.

Authors' code optimizes the network weights and architecture weights alternatively in mini-batches. 还进一步探讨了使用二阶优化（unroll）来替代一阶，来提高性能的可能性。

NNI 的实现基于[官方实现](https://github.com/quark0/darts)以及一个[第三方实现](https://github.com/khanrc/pt.darts)。 DARTS on NNI is designed to be general for arbitrary search space. A CNN search space tailored for CIFAR10, same as the original paper, is implemented as a use case of DARTS.

## Reproduction Results

The above-mentioned example is meant to reproduce the results in the paper, we do experiments with first and second order optimization. 由于时间限制，我们仅从第二阶段重新训练了*一次**最佳架构*。 我们的结果目前与论文的结果相当。 稍后会增加更多结果

|              | 论文中           | 重现   |
| ------------ | ------------- | ---- |
| 一阶 (CIFAR10) | 3.00 +/- 0.14 | 2.78 |
| 二阶（CIFAR10）  | 2.76 +/- 0.09 | 2.89 |

## Examples

### CNN Search Space

[Example code](https://github.com/microsoft/nni/tree/master/examples/nas/darts)

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/darts
python3 search.py

# train the best architecture
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_49.json
```

## Reference

### PyTorch

```eval_rst
..  autoclass:: nni.nas.pytorch.darts.DartsTrainer
    :members:

    .. automethod:: __init__

..  autoclass:: nni.nas.pytorch.darts.DartsMutator
    :members:
```
