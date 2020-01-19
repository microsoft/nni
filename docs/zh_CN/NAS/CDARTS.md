# CDARTS

## 介绍

CDARTS 在搜索和评估网络之间构建了循环反馈机制。 首先，搜索网络会生成初始结构用于评估，以便优化评估网络的权重。 然后，通过分类中通过的标签，以及评估网络中特征蒸馏的正则化来进一步优化搜索网络中的架构。 重复上述循环来优化搜索和评估网路，从而使结构得到训练，成为最终的评估网络。

在 `CdartsTrainer` 的实现中，首先分别实例化了两个 Model 和 Mutator。 第一个 Model 被称为"搜索网络"，使用 `RegularizedDartsMutator` 来进行变化。它与 `DartsMutator` 稍有差别。 The second model is the "evaluation network", which is mutated with a discrete mutator that leverages the previous search network mutator, to sample a single path each time. Trainers train models and mutators alternatively. Users can refer to [references](#reference) if they are interested in more details on these trainers and mutators.

## Reproduction Results

This is CDARTS based on the NNI platform, which currently supports CIFAR10 search and retrain. ImageNet search and retrain should also be supported, and we provide corresponding interfaces. Our reproduced results on NNI are slightly lower than the paper, but much higher than the original DARTS. Here we show the results of three independent experiments on CIFAR10.

| Runs | Paper |  NNI  |
| ---- |:-----:|:-----:|
| 1    | 97.52 | 97.44 |
| 2    | 97.53 | 97.48 |
| 3    | 97.58 | 97.56 |


## Examples

[Example code](https://github.com/microsoft/nni/tree/master/examples/nas/cdarts)

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# install apex for distributed training.
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext

# search the best architecture
cd examples/nas/cdarts
bash run_search_cifar.sh

# train the best architecture.
bash run_retrain_cifar.sh
```

## Reference

### PyTorch

```eval_rst
..  autoclass:: nni.nas.pytorch.cdarts.CdartsTrainer
    :members:

    .. automethod:: __init__

..  autoclass:: nni.nas.pytorch.cdarts.RegularizedDartsMutator
    :members:

..  autoclass:: nni.nas.pytorch.cdarts.DartsDiscreteMutator
    :members:

    .. automethod:: __init__

..  autoclass:: nni.nas.pytorch.cdarts.RegularizedMutatorParallel
    :members:
```
