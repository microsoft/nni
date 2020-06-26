# NAS Benchmark (experimental)

## Prerequisites

* To avoid storage and legal issues, we do not provide any generated databases. Users have to generate the database on your own. All generation scripts can be found in `examples/nas/benchmark`. To ease the burden of installing multiple dependencies, we strongly recommend users to use docker to run the generation scripts.
* Please prepare a folder to household all the benchmark databases. By default, it can be found at `${HOME}/.nni/nasbenchmark`. You can place it anywhere you like, and specify it in `NASBENCHMARK_DIR` before importing NNI.
* Please install `peewee` by `pip install peewee`, which NNI uses to connect to database.

## NAS-Bench-101

[Paper link](https://arxiv.org/abs/1902.09635) [Open-source](https://github.com/google-research/nasbench)

NAS-Bench-101 contains 423,624 unique neural networks, combined with 4 variations in number of epochs (4, 12, 36, 108), each of which is trained 3 times. It is a cell-wise search space, which constructs and stacks a cell by enumerating DAGs with at most 7 operators, and no more than 9 connections. All operators can be chosen from `CONV3X3_BN_RELU`, `CONV1X1_BN_RELU` and `MAXPOOL3X3`, except the first operator (always `INPUT`) and last operator (always `OUTPUT`).

Notably, NAS-Bench-101 eliminates invalid cells (e.g., there is no path from input to output, or there is redundant computation). Furthermore, isomorphic cells are de-duplicated, i.e., all the remaining cells are computationally unique.

### Preparation

```bash
docker run -e NNI_VERSION=${NNI_VERSION} -v ${HOME}/.nni/nasbenchmark:/outputs .:/root tensorflow/tensorflow:1.15.2-py3 /bin/bash /root/nasbench101.sh
```

Please replace `${NNI_VERSION}` with any NNI version, for example, v1.6 or master.

It takes about 70 minutes to dump the records and build index. Output size is about 1.9GB.

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

[Paper link](https://arxiv.org/abs/2001.00326) [Open-source API](https://github.com/D-X-Y/NAS-Bench-201) [Implementations](https://github.com/D-X-Y/AutoDL-Projects)

NAS-Bench-201 is a cell-wise search space that views nodes as tensors and edges as operators. The search space contains all possible densely-connected DAGs with 4 nodes, resulting in 15,625 candidates in total. Each operator (i.e., edge) is selected from a pre-defined operator set (`NONE`, `SKIP_CONNECT`, `CONV_1X1`, `CONV_3X3` and `AVG_POOL_3X3`). Training appraoches vary in the dataset used (CIFAR-10, CIFAR-100, ImageNet) and number of epochs scheduled (12 and 200). Each combination of architecture and training approach is repeated 1 - 3 times with different random seeds.

### Preparation

```bash
docker run -e NNI_VERSION=${NNI_VERSION} -v ${HOME}/.nni/nasbenchmark:/outputs .:/root ufoym/deepo:torch-cpu /bin/bash /root/nasbench201.sh
```

Please replace `${NNI_VERSION}` with any NNI version, for example, v1.6 or master.

The process takes about several minutes to download (~4GB) depending on the network and about 80 minutes to convert. The expected database size is ~2.7GB.

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

[Paper link](https://arxiv.org/abs/1905.13214) [Open-source](https://github.com/facebookresearch/nds)

_On Network Design Spaces for Visual Recognition_ released computed statistics of over 100,000 configurations (models + hyper-parameters) sampled from multiple model families, including vanilla (feedforward network loosely inspired by VGG), ResNet and ResNeXt (residual basic block and residual bottleneck block) and NAS cells (following popular design from NASNet, Ameoba, PNAS, ENAS and DARTS). Most configurations are trained only once with a fixed seed, except a few that are trained twice or three times.

Instead of storing results obtained with different configurations in separate files, we dump them into one single database to enable comparison in multiple dimensions. Specifically, we use `model_family` to distinguish model types, `model_spec` for all hyper-parameters needed to build this model, `cell_spec` for detailed information on operators and connections if it is a NAS cell, `generator` to denote the sampling policy through which this configuration is generated. Refer to API documentation for details.

### Preparation

```bash
docker run -e NNI_VERSION=${NNI_VERSION} -v ${HOME}/.nni/nasbenchmark:/outputs .:/root python:3.8 /bin/bash /root/nds.sh
```

Please replace `${NNI_VERSION}` with any NNI version, for example, v1.6 or master.

The conversion takes around 80 minutes to complete. Output size is about 1.6GB.

### API Documentation

TODO
