# CDARTS

## 介绍

CDARTS 在搜索和评估网络之间构建了循环反馈机制。 首先，搜索网络会生成初始结构用于评估，以便优化评估网络的权重。 然后，通过分类中通过的标签，以及评估网络中特征蒸馏的正则化来进一步优化搜索网络中的架构。 重复上述循环来优化搜索和评估网路，从而使结构得到训练，成为最终的评估网络。

在 `CdartsTrainer` 的实现中，首先分别实例化了两个 Model 和 Mutator。 第一个 Model 被称为"搜索网络"，使用 `RegularizedDartsMutator` 来进行变化。它与 `DartsMutator` 稍有差别。 第二个 Model 是“评估网络”，它里用前面搜索网络的 Mutator 来创建了一个离散的 Mutator，来每次采样一条路径。 Trainer 会交替训练 Model 和 Mutator。 如果对 Trainer 和 Mutator 的实现感兴趣，可参考[这里](#reference)。

## 重现结果

这是基于 NNI 平台的 CDARTS，该平台目前支持 CIFAR10 搜索和重新训练。 同时也支持 ImageNet 的搜索和重新训练，并有相应的接口。 在 NNI 上重现的结果略低于论文，但远高于原始 DARTS。 这里展示了在 CIFAR10 上的三个独立实验的结果。

| 运行 |  论文   |  NNI  |
| -- |:-----:|:-----:|
| 1  | 97.52 | 97.44 |
| 2  | 97.53 | 97.48 |
| 3  | 97.58 | 97.56 |


## 示例

[示例代码](https://github.com/microsoft/nni/tree/master/examples/nas/cdarts)

```bash
＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
git clone https://github.com/Microsoft/nni.git

# 为分布式训练安装 apex
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext

# 搜索最好的架构
cd examples/nas/cdarts
bash run_search_cifar.sh

# 训练最好的架构
bash run_retrain_cifar.sh
```

## 参考

### PyTorch

```eval_rst
..  autoclass:: nni.nas.pytorch.cdarts.CdartsTrainer
    :members:

..  autoclass:: nni.nas.pytorch.cdarts.RegularizedDartsMutator
    :members:

..  autoclass:: nni.nas.pytorch.cdarts.DartsDiscreteMutator
    :members:

..  autoclass:: nni.nas.pytorch.cdarts.RegularizedMutatorParallel
    :members:
```
