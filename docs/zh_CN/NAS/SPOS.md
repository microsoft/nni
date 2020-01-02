# 单路径 One-Shot (SPOS)

## 介绍

在 [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) 中提出的 one-shot NAS 方法，通过构造简化的通过统一路径采样方法训练的超网络来解决 One-Shot 模型训练的问题。这样所有架构（及其权重）都得到了完全且平等的训练。 然后，采用进化算法无需任何微调即可有效的搜索出性能最佳的体系结构。

在 NNI 上的实现基于 [官方 Repo](https://github.com/megvii-model/SinglePathOneShot). 实现了一个训练超级网络的 Trainer，以及一个利用 NNI 框架能力来加速进化搜索阶段的进化 Tuner。 还展示了

## 示例

此示例是论文中的搜索空间，使用 flops 限制来执行统一的采样方法。

[示例代码](https://github.com/microsoft/nni/tree/master/examples/nas/spos)

### 必需组件

由于使用了 DALI 来加速 ImageNet 的数据读取，需要 NVIDIA DALI >= 0.16。 [安装指南](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html)

从[这里](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aHVppN) (由 [Megvii](https://github.com/megvii-model) 维护) 下载 flops 查找表。 将 `op_flops_dict.pkl` 和 `checkpoint-150000.pth.tar` (如果不需要重新训练超网络) 放到 `data` 目录中。

准备标准格式的 ImageNet (参考[这里的脚本](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4))。 将其链接到 `data/imagenet` 会更方便。

准备好后，应具有以下代码结构：

```
spos
├── architecture_final.json
├── blocks.py
├── config_search.yml
├── data
│   ├── imagenet
│   │   ├── train
│   │   └── val
│   └── op_flops_dict.pkl
├── dataloader.py
├── network.py
├── readme.md
├── scratch.py
├── supernet.py
├── tester.py
├── tuner.py
└── utils.py
```

### 步骤 1. 训练超网络

```
python supernet.py
```

会将检查点导出到 `checkpoints` 目录中，为下一步做准备。

NOTE: The data loading used in the official repo is [slightly different from usual](https://github.com/megvii-model/SinglePathOneShot/issues/5), as they use BGR tensor and keep the values between 0 and 255 intentionally to align with their own DL framework. The option `--spos-preprocessing` will simulate the behavior used originally and enable you to use the checkpoints pretrained.

### Step 2. Evolution Search

Single Path One-Shot leverages evolution algorithm to search for the best architecture. The tester, which is responsible for testing the sampled architecture, recalculates all the batch norm for a subset of training images, and evaluates the architecture on the full validation set.

In order to make the tuner aware of the flops limit and have the ability to calculate the flops, we created a new tuner called `EvolutionWithFlops` in `tuner.py`, inheriting the tuner in SDK.

To have a search space ready for NNI framework, first run

```
nnictl ss_gen -t "python tester.py"
```

This will generate a file called `nni_auto_gen_search_space.json`, which is a serialized representation of your search space.

By default, it will use `checkpoint-150000.pth.tar` downloaded previously. In case you want to use the checkpoint trained by yourself from the last step, specify `--checkpoint` in the command in `config_search.yml`.

Then search with evolution tuner.

```
nnictl create --config config_search.yml
```

The final architecture exported from every epoch of evolution can be found in `checkpoints` under the working directory of your tuner, which, by default, is `$HOME/nni/experiments/your_experiment_id/log`.

### Step 3. Train from Scratch

```
python scratch.py
```

By default, it will use `architecture_final.json`. This architecture is provided by the official repo (converted into NNI format). You can use any architecture (e.g., the architecture found in step 2) with `--fixed-arc` option.

## Reference

### PyTorch

```eval_rst
..  autoclass:: nni.nas.pytorch.spos.SPOSEvolution
    :members:

    .. automethod:: __init__

..  autoclass:: nni.nas.pytorch.spos.SPOSSupernetTrainer
    :members:

    .. automethod:: __init__

..  autoclass:: nni.nas.pytorch.spos.SPOSSupernetTrainingMutator
    :members:

    .. automethod:: __init__
```

## Known Limitations

* Block search only. Channel search is not supported yet.
* Only GPU version is provided here.

## Current Reproduction Results

Reproduction is still undergoing. Due to the gap between official release and original paper, we compare our current results with official repo (our run) and paper.

* Evolution phase is almost aligned with official repo. Our evolution algorithm shows a converging trend and reaches ~65% accuracy at the end of search. Nevertheless, this result is not on par with paper. For details, please refer to [this issue](https://github.com/megvii-model/SinglePathOneShot/issues/6).
* Retrain phase is not aligned. Our retraining code, which uses the architecture released by the authors, reaches 72.14% accuracy, still having a gap towards 73.61% by official release and 74.3% reported in original paper.
