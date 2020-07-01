# NAS 基准测试（测试版）

```eval_rst
..  toctree::
    :hidden:

    用法示例 <BenchmarksExample>
```

## 先决条件

* 准备目录来保存基准测试的数据库。 默认情况下，目录为 `${HOME}/.nni/nasbenchmark`。 可将其设置为任何位置，并在 import nni 前，通过 `NASBENCHMARK_DIR` 指定。
* 通过 `pip install peewee` 命令安装 `peewee`，NNI 用其连接数据库。

## 准备数据

为了避免存储和法规问题，NNI 不提供数据库。 强烈建议通过 Docker 来运行生成的脚本，减少安装依赖项的时间。 步骤：

**步骤 1.** 克隆 NNI 存储库。 将 `${NNI_VERSION}` 替换为发布的版本或分支名称，例如：`v1.6`。

```bash
git clone -b ${NNI_VERSION} https://github.com/microsoft/nni
```

**步骤 2.** 运行 Docker。

对于 NAS-Bench-101,

```bash
docker run -v ${HOME}/.nni/nasbenchmark:/outputs -v /path/to/your/nni:/nni tensorflow/tensorflow:1.15.2-py3 /bin/bash /nni/examples/nas/benchmarks/nasbench101.sh
```

对于 NAS-Bench-201,

```bash
docker run -v ${HOME}/.nni/nasbenchmark:/outputs -v /path/to/your/nni:/nni ufoym/deepo:pytorch-cpu /bin/bash /nni/examples/nas/benchmarks/nasbench201.sh
```

对于 NDS,

```bash
docker run -v ${HOME}/.nni/nasbenchmark:/outputs -v /path/to/your/nni:/nni python:3.7 /bin/bash /nni/examples/nas/benchmarks/nds.sh
```

确保至少有 10GB 的可用磁盘空间，运行过程可能需要几个小时。

## 示例用法

参考[基准测试 API 的用法](./BenchmarksExample)。

## NAS-Bench-101

[论文](https://arxiv.org/abs/1902.09635) &nbsp; &nbsp; [代码](https://github.com/google-research/nasbench)

NAS-Bench-101 contains 423,624 unique neural networks, combined with 4 variations in number of epochs (4, 12, 36, 108), each of which is trained 3 times. It is a cell-wise search space, which constructs and stacks a cell by enumerating DAGs with at most 7 operators, and no more than 9 connections. All operators can be chosen from `CONV3X3_BN_RELU`, `CONV1X1_BN_RELU` and `MAXPOOL3X3`, except the first operator (always `INPUT`) and last operator (always `OUTPUT`).

Notably, NAS-Bench-101 eliminates invalid cells (e.g., there is no path from input to output, or there is redundant computation). Furthermore, isomorphic cells are de-duplicated, i.e., all the remaining cells are computationally unique.

### API Documentation

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

[Paper link](https://arxiv.org/abs/2001.00326) &nbsp; &nbsp; [Open-source API](https://github.com/D-X-Y/NAS-Bench-201) &nbsp; &nbsp;[Implementations](https://github.com/D-X-Y/AutoDL-Projects)

NAS-Bench-201 is a cell-wise search space that views nodes as tensors and edges as operators. The search space contains all possible densely-connected DAGs with 4 nodes, resulting in 15,625 candidates in total. Each operator (i.e., edge) is selected from a pre-defined operator set (`NONE`, `SKIP_CONNECT`, `CONV_1X1`, `CONV_3X3` and `AVG_POOL_3X3`). Training appraoches vary in the dataset used (CIFAR-10, CIFAR-100, ImageNet) and number of epochs scheduled (12 and 200). Each combination of architecture and training approach is repeated 1 - 3 times with different random seeds.

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

[Paper link](https://arxiv.org/abs/1905.13214) &nbsp; &nbsp; [Open-source](https://github.com/facebookresearch/nds)

_On Network Design Spaces for Visual Recognition_ released trial statistics of over 100,000 configurations (models + hyper-parameters) sampled from multiple model families, including vanilla (feedforward network loosely inspired by VGG), ResNet and ResNeXt (residual basic block and residual bottleneck block) and NAS cells (following popular design from NASNet, Ameoba, PNAS, ENAS and DARTS). Most configurations are trained only once with a fixed seed, except a few that are trained twice or three times.

Instead of storing results obtained with different configurations in separate files, we dump them into one single database to enable comparison in multiple dimensions. Specifically, we use `model_family` to distinguish model types, `model_spec` for all hyper-parameters needed to build this model, `cell_spec` for detailed information on operators and connections if it is a NAS cell, `generator` to denote the sampling policy through which this configuration is generated. Refer to API documentation for details.

## Available Operators

Here is a list of available operators used in NDS.

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

### API Documentation

```eval_rst
.. autofunction:: nni.nas.benchmarks.nds.query_nds_trial_stats

.. autoclass:: nni.nas.benchmarks.nds.NdsTrialConfig

.. autoclass:: nni.nas.benchmarks.nds.NdsTrialStats

.. autoclass:: nni.nas.benchmarks.nds.NdsIntermediateStats
```