# NAS 基准测试（测试版）

```eval_rst
..  toctree::
    :hidden:

    用法示例 <BenchmarksExample>
```

## Introduction
To imporve the reproducibility of NAS algorithms as well as reducing computing resource requirements, researchers proposed a series of NAS benchmarks such as [NAS-Bench-101](https://arxiv.org/abs/1902.09635), [NAS-Bench-201](https://arxiv.org/abs/2001.00326), [NDS](https://arxiv.org/abs/1905.13214), etc. NNI provides a query interface for users to acquire these benchmarks. Within just a few lines of code, researcher are able to evaluate their NAS algorithms easily and fairly by utilizing these benchmarks.

## 先决条件

* 准备目录来保存基准测试的数据库。 默认情况下，目录为 `${HOME}/.nni/nasbenchmark`。 可将其设置为任何位置，并在 import nni 前，通过 `NASBENCHMARK_DIR` 指定。
* 通过 `pip install peewee` 命令安装 `peewee`，NNI 用其连接数据库。

## 准备数据

To avoid storage and legality issues, we do not provide any prepared databases. 步骤：

1. Clone NNI to your machine and enter `examples/nas/benchmarks` directory.
```
git clone -b ${NNI_VERSION} https://github.com/microsoft/nni
cd nni/examples/nas/benchmarks
```
Replace `${NNI_VERSION}` with a released version name or branch name, e.g., `v1.7`.

2. Install dependencies via `pip3 install -r xxx.requirements.txt`. `xxx` can be `nasbench101`, `nasbench201` or `nds`.
3. Generate the database via `./xxx.sh`. The directory that stores the benchmark file can be configured with `NASBENCHMARK_DIR` environment variable, which defaults to `~/.nni/nasbenchmark`. Note that the NAS-Bench-201 dataset will be downloaded from a google drive.

确保至少有 10GB 的可用磁盘空间，运行过程可能需要几个小时。

## 示例用法

参考[基准测试 API 的用法](./BenchmarksExample)。

## NAS-Bench-101

[论文](https://arxiv.org/abs/1902.09635) &nbsp; &nbsp; [代码](https://github.com/google-research/nasbench)

NAS-Bench-101 包含 423,624 个独立的神经网络，再加上 4 个 Epoch (4, 12, 36, 108) 时的变化，以及每个都要训练 3 次。 这是基于 Cell 的搜索空间，通过枚举最多 7 个有向图的运算符来构造并堆叠 Cell，连接数量不超过 9 个。 除了第一个 (必须为 `INPUT`) 和最后一个运算符 (必须为 `OUTPUT`)，可选的运算符有 `CONV3X3_BN_RELU`, `CONV1X1_BN_RELU` 和 `MAXPOOL3X3`。

注意，NAS-Bench-101 消除了非法的 Cell（如，从输入到输出没有路径，或存在冗余的计算）。 此外，同构的 Cell 会被去掉，即，所有的 Cell 从计算上看是一致的。

### API 文档

```eval_rst
.. autofunction:: nni.nas.benchmarks.nasbench101.query_nb101_trial_stats

.. autoattribute:: nni.nas.benchmarks.nasbench101.INPUT

.. autoattribute:: nni.nas.benchmarks.nasbench101.OUTPUT

.. autoattribute:: nni.nas.benchmarks.nasbench101.CONV3X3_BN_RELU

.. autoattribute:: nni.nas.benchmarks.nasbench101.CONV1X1_BN_RELU

.. autoattribute:: nni.nas.benchmarks.nasbench101.MAXPOOL3X3

.. autoclass:: nni.nas.benchmarks.nasbench101.Nb101TrialConfig

.. autoclass:: nni.nas.benchmarks.nasbench101.Nb101TrialStats

.. autoclass:: nni.nas.benchmarks.nasbench101.Nb101IntermediateStats

.. autofunction:: nni.nas.benchmarks.nasbench101.graph_util.nasbench_format_to_architecture_repr

.. autofunction:: nni.nas.benchmarks.nasbench101.graph_util.infer_num_vertices

.. autofunction:: nni.nas.benchmarks.nasbench101.graph_util.hash_module
```

## NAS-Bench-201

[论文](https://arxiv.org/abs/2001.00326) &nbsp; &nbsp; [API](https://github.com/D-X-Y/NAS-Bench-201) &nbsp; &nbsp;[实现](https://github.com/D-X-Y/AutoDL-Projects)

NAS-Bench-201 是单元格的搜索空间，并将张量当作节点，运算符当作边。 搜索空间包含了 4 个节点所有密集连接的有向图，共有 15,625 个候选项。 每个运算符（即：边）从预定义的运算符集中选择 (`NONE`, `SKIP_CONNECT`, `CONV_1X1`, `CONV_3X3` 和 `AVG_POOL_3X3`)。 训练方法根据数据集 (CIFAR-10, CIFAR-100, ImageNet) 和 Epoch 数量 (12 和 200)，而有所不同。 每个架构和训练方法的组合会随机重复 1 到 3 次。

### API 文档


```eval_rst
.. autofunction:: nni.nas.benchmarks.nasbench201.query_nb201_trial_stats

.. autoattribute:: nni.nas.benchmarks.nasbench201.NONE

.. autoattribute:: nni.nas.benchmarks.nasbench201.SKIP_CONNECT

.. autoattribute:: nni.nas.benchmarks.nasbench201.CONV_1X1

.. autoattribute:: nni.nas.benchmarks.nasbench201.CONV_3X3

.. autoattribute:: nni.nas.benchmarks.nasbench201.AVG_POOL_3X3

.. autoclass:: nni.nas.benchmarks.nasbench201.Nb201TrialConfig

.. autoclass:: nni.nas.benchmarks.nasbench201.Nb201TrialStats

.. autoclass:: nni.nas.benchmarks.nasbench201.Nb201IntermediateStats
```

## NDS

[论文](https://arxiv.org/abs/1905.13214) &nbsp; &nbsp; [代码](https://github.com/facebookresearch/nds)

_On Network Design Spaces for Visual Recognition_ 发布了来自多个模型系列，超过 100,000 个配置（模型加超参组合）的统计，包括 vanilla (受 VGG 启发的松散前馈网络), ResNet 和 ResNeXt (残差基本模块和残差瓶颈模块) 以及 NAS 单元格 (遵循 NASNet, Ameoba, PNAS, ENAS 和 DARTS 的设计)。 大部分配置只采用固定的随机种子训练一次，但少部分会训练两到三次。

NNI 会将不同配置的结果存到单个数据库中，而不是单独的文件中，以便从各个维度进行比较。 在实现上，`model_family` 用来保存模型类型，`model_spec` 用来保存构建模型所需的参数，在使用 NAS 时，`cell_spec` 保存运算符和连接的详细信息，`generator` 表示配置生成的采样策略。 详情可参考 API 文档。

## 可用的运算符

NDS 中可用的运算符列表。

```eval_rst
.. autoattribute:: nni.nas.benchmarks.nds.constants.NONE

.. autoattribute:: nni.nas.benchmarks.nds.constants.SKIP_CONNECT

.. autoattribute:: nni.nas.benchmarks.nds.constants.AVG_POOL_3X3

.. autoattribute:: nni.nas.benchmarks.nds.constants.MAX_POOL_3X3

.. autoattribute:: nni.nas.benchmarks.nds.constants.MAX_POOL_5X5

.. autoattribute:: nni.nas.benchmarks.nds.constants.MAX_POOL_7X7

.. autoattribute:: nni.nas.benchmarks.nds.constants.CONV_1X1

.. autoattribute:: nni.nas.benchmarks.nds.constants.CONV_3X3

.. autoattribute:: nni.nas.benchmarks.nds.constants.CONV_3X1_1X3

.. autoattribute:: nni.nas.benchmarks.nds.constants.CONV_7X1_1X7

.. autoattribute:: nni.nas.benchmarks.nds.constants.DIL_CONV_3X3

.. autoattribute:: nni.nas.benchmarks.nds.constants.DIL_CONV_5X5

.. autoattribute:: nni.nas.benchmarks.nds.constants.SEP_CONV_3X3

.. autoattribute:: nni.nas.benchmarks.nds.constants.SEP_CONV_5X5

.. autoattribute:: nni.nas.benchmarks.nds.constants.SEP_CONV_7X7

.. autoattribute:: nni.nas.benchmarks.nds.constants.DIL_SEP_CONV_3X3
```

### API 文档

```eval_rst
.. autofunction:: nni.nas.benchmarks.nds.query_nds_trial_stats

.. autoclass:: nni.nas.benchmarks.nds.NdsTrialConfig

.. autoclass:: nni.nas.benchmarks.nds.NdsTrialStats

.. autoclass:: nni.nas.benchmarks.nds.NdsIntermediateStats
```
