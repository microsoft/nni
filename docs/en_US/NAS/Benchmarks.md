# NAS Benchmarks (experimental)

```eval_rst
..  toctree::
    :hidden:

    Example Usages <BenchmarksExample>
```

## Introduction
To imporve the reproducibility of NAS algorithms as well as reducing computing resource requirements, researchers proposed a series of NAS benchmarks such as [NAS-Bench-101](https://arxiv.org/abs/1902.09635), [NAS-Bench-201](https://arxiv.org/abs/2001.00326), [NDS](https://arxiv.org/abs/1905.13214), etc. NNI provides a query interface for users to acquire these benchmarks. Within just a few lines of code, researcher are able to evaluate their NAS algorithms easily and fairly by utilizing these benchmarks.

## Prerequisites

* Please prepare a folder to household all the benchmark databases. By default, it can be found at `${HOME}/.nni/nasbenchmark`. You can place it anywhere you like, and specify it in `NASBENCHMARK_DIR` before importing NNI.
* Please install `peewee` via `pip install peewee`, which NNI uses to connect to database.

## Data Preparation

To avoid storage and legality issues, we do not provide any prepared databases. Please follow the following steps.

1. Clone NNI to your machine and enter `examples/nas/benchmarks` directory.
```
git clone -b ${NNI_VERSION} https://github.com/microsoft/nni
cd nni/examples/nas/benchmarks
```
Replace `${NNI_VERSION}` with a released version name or branch name, e.g., `v1.7`.

2. Install dependencies via `pip3 install -r xxx.requirements.txt`. `xxx` can be `nasbench101`, `nasbench201` or `nds`.
3. Generate the database via `./xxx.sh`. The directory that stores the benchmark file can be configured with `NASBENCHMARK_DIR` environment variable, which defaults to `~/.nni/nasbenchmark`. Note that the NAS-Bench-201 dataset will be downloaded from a google drive.

Please make sure there is at least 10GB free disk space and note that the conversion process can take up to hours to complete.

## Example Usages

Please refer to [examples usages of Benchmarks API](./BenchmarksExample).

## NAS-Bench-101

[Paper link](https://arxiv.org/abs/1902.09635) &nbsp; &nbsp; [Open-source](https://github.com/google-research/nasbench)

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

### API Documentation


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
