# NAS Benchmark (experimental)

## Prerequisites

* Please prepare a folder to household all the benchmark databases. By default, it can be found at `${HOME}/.nni/nasbenchmark`. You can place it anywhere you like, and specify it in `NASBENCHMARK_DIR` before importing NNI.
* Please install `peewee` via `pip install peewee`, which NNI uses to connect to database.

## Data Preparation

To avoid storage and legal issues, we do not provide any prepared databases. We strongly recommend users to use docker to run the generation scripts, to ease the burden of installing multiple dependencies. Please follow the following steps.

1. Clone NNI repo. Replace `${NNI_VERSION}` with a released version name or branch name, e.g., `v1.6`.

```bash
git clone -b ${NNI_VERSION} https://github.com/microsoft/nni
```

2. Run docker.

For NAS-Bench-101,

```bash
docker run -v ${HOME}/.nni/nasbenchmark:/outputs -v /path/to/your/nni:/nni tensorflow/tensorflow:1.15.2-py3 /bin/bash /nni/examples/nas/benchmark/nasbench101.sh
```

For NAS-Bench-201,

```bash
docker run -v ${HOME}/.nni/nasbenchmark:/outputs -v /path/to/your/nni:/nni ufoym/deepo:torch-cpu /bin/bash /nni/examples/nas/benchmark/nasbench201.sh
```

For NDS,

```bash
docker run -v ${HOME}/.nni/nasbenchmark:/outputs -v /path/to/your/nni:/nni python:3.7 /bin/bash /nni/examples/nas/benchmark/nds.sh
```

Please make sure there is at least 10GB free disk space and note that the conversion process can take up to hours to complete.

## NAS-Bench-101

[Paper link](https://arxiv.org/abs/1902.09635) &nbsp; &nbsp; [Open-source](https://github.com/google-research/nasbench)

NAS-Bench-101 contains 423,624 unique neural networks, combined with 4 variations in number of epochs (4, 12, 36, 108), each of which is trained 3 times. It is a cell-wise search space, which constructs and stacks a cell by enumerating DAGs with at most 7 operators, and no more than 9 connections. All operators can be chosen from `CONV3X3_BN_RELU`, `CONV1X1_BN_RELU` and `MAXPOOL3X3`, except the first operator (always `INPUT`) and last operator (always `OUTPUT`).

Notably, NAS-Bench-101 eliminates invalid cells (e.g., there is no path from input to output, or there is redundant computation). Furthermore, isomorphic cells are de-duplicated, i.e., all the remaining cells are computationally unique.

### API Documentation

```eval_rst
.. autofunction:: nni.nas.benchmark.nasbench101.query_nb101_computed_stats

.. autoattribute:: nni.nas.benchmark.nasbench101.INPUT

.. autoattribute:: nni.nas.benchmark.nasbench101.OUTPUT

.. autoattribute:: nni.nas.benchmark.nasbench101.CONV3X3_BN_RELU

.. autoattribute:: nni.nas.benchmark.nasbench101.CONV1X1_BN_RELU

.. autoattribute:: nni.nas.benchmark.nasbench101.MAXPOOL3X3

.. autoclass:: nni.nas.benchmark.nasbench101.Nb101RunConfig

.. autoclass:: nni.nas.benchmark.nasbench101.Nb101ComputedStats

.. autoclass:: nni.nas.benchmark.nasbench101.Nb101IntermediateStats

.. autofunction:: nni.nas.benchmark.nasbench101.graph_util.nasbench_format_to_architecture_repr

.. autofunction:: nni.nas.benchmark.nasbench101.graph_util.infer_num_vertices

.. autofunction:: nni.nas.benchmark.nasbench101.graph_util.hash_module
```

## NAS-Bench-201

[Paper link](https://arxiv.org/abs/2001.00326) &nbsp; &nbsp; [Open-source API](https://github.com/D-X-Y/NAS-Bench-201) &nbsp; &nbsp;[Implementations](https://github.com/D-X-Y/AutoDL-Projects)

NAS-Bench-201 is a cell-wise search space that views nodes as tensors and edges as operators. The search space contains all possible densely-connected DAGs with 4 nodes, resulting in 15,625 candidates in total. Each operator (i.e., edge) is selected from a pre-defined operator set (`NONE`, `SKIP_CONNECT`, `CONV_1X1`, `CONV_3X3` and `AVG_POOL_3X3`). Training appraoches vary in the dataset used (CIFAR-10, CIFAR-100, ImageNet) and number of epochs scheduled (12 and 200). Each combination of architecture and training approach is repeated 1 - 3 times with different random seeds.

### API Documentation


```eval_rst
.. autofunction:: nni.nas.benchmark.nasbench201.query_nb201_computed_stats

.. autoattribute:: nni.nas.benchmark.nasbench201.NONE

.. autoattribute:: nni.nas.benchmark.nasbench201.SKIP_CONNECT

.. autoattribute:: nni.nas.benchmark.nasbench201.CONV_1X1

.. autoattribute:: nni.nas.benchmark.nasbench201.CONV_3X3

.. autoattribute:: nni.nas.benchmark.nasbench201.AVG_POOL_3X3

.. autoclass:: nni.nas.benchmark.nasbench201.Nb201RunConfig

.. autoclass:: nni.nas.benchmark.nasbench201.Nb201ComputedStats

.. autoclass:: nni.nas.benchmark.nasbench201.Nb201IntermediateStats
```

## NDS

[Paper link](https://arxiv.org/abs/1905.13214) &nbsp; &nbsp; [Open-source](https://github.com/facebookresearch/nds)

_On Network Design Spaces for Visual Recognition_ released computed statistics of over 100,000 configurations (models + hyper-parameters) sampled from multiple model families, including vanilla (feedforward network loosely inspired by VGG), ResNet and ResNeXt (residual basic block and residual bottleneck block) and NAS cells (following popular design from NASNet, Ameoba, PNAS, ENAS and DARTS). Most configurations are trained only once with a fixed seed, except a few that are trained twice or three times.

Instead of storing results obtained with different configurations in separate files, we dump them into one single database to enable comparison in multiple dimensions. Specifically, we use `model_family` to distinguish model types, `model_spec` for all hyper-parameters needed to build this model, `cell_spec` for detailed information on operators and connections if it is a NAS cell, `generator` to denote the sampling policy through which this configuration is generated. Refer to API documentation for details.

## Available Operators

Here is a list of available operators used in NDS.

```eval_rst
.. autoattribute:: nni.nas.benchmark.nds.constants.NONE

.. autoattribute:: nni.nas.benchmark.nds.constants.SKIP_CONNECT

.. autoattribute:: nni.nas.benchmark.nds.constants.AVG_POOL_3X3

.. autoattribute:: nni.nas.benchmark.nds.constants.MAX_POOL_3X3

.. autoattribute:: nni.nas.benchmark.nds.constants.MAX_POOL_5X5

.. autoattribute:: nni.nas.benchmark.nds.constants.MAX_POOL_7X7

.. autoattribute:: nni.nas.benchmark.nds.constants.CONV_1X1

.. autoattribute:: nni.nas.benchmark.nds.constants.CONV_3X3

.. autoattribute:: nni.nas.benchmark.nds.constants.CONV_3X1_1X3

.. autoattribute:: nni.nas.benchmark.nds.constants.CONV_7X1_1X7

.. autoattribute:: nni.nas.benchmark.nds.constants.DIL_CONV_3X3

.. autoattribute:: nni.nas.benchmark.nds.constants.DIL_CONV_5X5

.. autoattribute:: nni.nas.benchmark.nds.constants.SEP_CONV_3X3

.. autoattribute:: nni.nas.benchmark.nds.constants.SEP_CONV_5X5

.. autoattribute:: nni.nas.benchmark.nds.constants.SEP_CONV_7X7

.. autoattribute:: nni.nas.benchmark.nds.constants.DIL_SEP_CONV_3X3
```

### API Documentation

```eval_rst
.. autofunction:: nni.nas.benchmark.nds.query_nds_computed_stats

.. autoclass:: nni.nas.benchmark.nds.NdsRunConfig

.. autoclass:: nni.nas.benchmark.nds.NdsComputedStats

.. autoclass:: nni.nas.benchmark.nds.NdsIntermediateStats
```