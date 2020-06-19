# Documentation Draft for NAS Benchmarks

This document will be moved to documentation folder later.

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


## NAS-Bench-201

### Preparation

```bash
docker run -e NNI_VERSION=${NNI_VERSION} -v ${HOME}/.nni/nasbenchmark:/outputs .:/root ufoym/deepo:torch-cpu /bin/bash /root/nasbench201.sh
```

Please replace `${NNI_VERSION}` with any NNI version, for example, v1.6 or master.

The process takes about several minutes to download (~4GB) depending on the network and about 80 minutes to convert. The expected database size is ~2.7GB.

### API Documentation

TODO
