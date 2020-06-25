# NAS Benchmark (experimental)

## Prerequisites

* To avoid storage and legal issues, we do not provide any generated databases. Users have to generate the database on your own. All generation scripts can be found in `examples/nas/benchmark`. To ease the burden of installing multiple dependencies, we strongly recommend users to use docker to run the generation scripts.
* Please prepare a folder to household all the benchmark databases. By default, it can be found at `${HOME}/.nni/nasbenchmark`. You can place it anywhere you like, and specify it in `NASBENCHMARK_DIR` before importing NNI.
* Please install `peewee` by `pip install peewee`, which NNI uses to connect to database.

## NAS-Bench-101

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

### Preparation

```bash
docker run -e NNI_VERSION=${NNI_VERSION} -v ${HOME}/.nni/nasbenchmark:/outputs .:/root ufoym/deepo:torch-cpu /bin/bash /root/nasbench201.sh
```

Please replace `${NNI_VERSION}` with any NNI version, for example, v1.6 or master.

The process takes about several minutes to download (~4GB) depending on the network and about 80 minutes to convert. The expected database size is ~2.7GB.

### API Documentation

TODO


## NDS

```bash
docker run -e NNI_VERSION=${NNI_VERSION} -v ${HOME}/.nni/nasbenchmark:/outputs .:/root python:3.8 /bin/bash /root/nds.sh
```

Please replace `${NNI_VERSION}` with any NNI version, for example, v1.6 or master.

The conversion takes around 80 minutes to complete. Output size is about 1.6GB.

### API Documentation

TODO
