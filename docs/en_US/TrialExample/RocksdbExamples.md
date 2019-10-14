# Tuning RocksDB on NNI

## Overview

[RocksDB](https://github.com/facebook/rocksdb) is a popular high performance embedded key-value database used in production systems at various web-scale enterprises including Facebook, Yahoo!, and LinkedIn.. It is a fork of [LevelDB](https://github.com/google/leveldb) by Facebook optimized to exploit many central processing unit (CPU) cores, and make efficient use of fast storage, such as solid-state drives (SSD), for input/output (I/O) bound workloads.

The performance of RocksDB is highly contingent on its tuning. However, because of the complexity of its underlying technology and a large number of configurable parameters, a good configuration is sometimes hard to obtain. NNI can help to address this issue. NNI supports many kinds of tuning algorithms to search the best configuration of RocksDB, and support many kinds of environments like local machine, remote servers and cloud. By following this example, you are able to search the best configuration of RocksDB for a `fillrandom` benchmark with SMAC and TPE tuners. Other tuners can be easily adopted in the same way. Please refer to [here](../Tuner/BuiltinTuner.md) for more information.

`code directory: example/trials/systems/rocksdb-fillrandom`

## Goals

This example illustrates how to use NNI to search the best configuration of RocksDB for a `fillrandom` benchmark supported by a benchmark tool `db_bench`, which is a official benchmark tool provided by RocksDB itself. Therefore, before running this example, please make sure NNI is installed and [`db_bench`](https://github.com/facebook/rocksdb/wiki/Benchmarking-tools) is in your `PATH`. Please refer to [here](../Tutorial/QuickStart.md) for detailed information about installation and preparing of NNI environment, and [here](https://github.com/facebook/rocksdb/blob/master/INSTALL.md) for compiling RocksDB as well as `db_bench`.

## Search Space

For simplicity, this example tunes three parameters, `buffer_size`, `min_write_buffer_num` and `level0_file_num_compaction_trigger`, for writing 16M keys with 20 Bytes of key size and 100 Bytes of value size randomly, based on writing operations per second (OPS). The search space is specified by a `search_space.json` file as shown below. Detailed explanation of search space could be found [here](https://github.com/microsoft/nni/blob/master/docs/en_US/Tutorial/SearchSpaceSpec.md).

```json
{
    "write_buffer_size": {
        "_type": "quniform",
        "_value": [2097152, 16777216, 1048576]
    },
    "min_write_buffer_number_to_merge": {
        "_type": "quniform",
        "_value": [2, 16, 1]
    },
    "level0_file_num_compaction_trigger": {
        "_type": "quniform",
        "_value": [2, 16, 1]
    }
}
```

`code directory: example/trials/systems/rocksdb-fillrandom/search_space.json`

## Benchmark

Benchmark code should receive a configuration from NNI manager, and report the corresponding benchmark result back. Following NNI APIs are designed for this purpose. In this example, writing operations per second (OPS) is used as a performance metric. Please refer to [here](Trials.md) for detailed information.

* Use `nni.get_next_parameter()` to get next system configuration.
* Use `nni.report_final_result(metric)` to report the benchmark result.

`code directory: example/trials/systems/rocksdb-fillrandom/main.py`

## Config

One could start a NNI experiment with a config file. Usually, a config file for NNI includes experiment settings (`trialConcurrency`, `maxExecDuration`, `maxTrialNum`, `trial gpuNum`, etc.), platform settings (`trainingServicePlatform`, etc.), path settings (`searchSpacePath`, `trial codeDir`, etc.) and tuner settings (`tuner`, `tuner optimize_mode`, etc.). Please refer to [here](../Tutorial/QuickStart.md) for more information.

Here is the example of tuning RocksDB with SMAC algorithm:

`code directory: examples/trials/systems/rocksdb-fillrandom/config_smac.yml`

Here is the example of tuning RocksDB with TPE algorithm:

`code directory: examples/trials/systems/rocksdb-fillrandom/config_tpe.yml`

## Launch the experiment

In order to run this example, you could enter the example folder and start the experiment using following commands:

```bash
# tuning RocksDB with SMAC tuner
nnictl create --config ./config_smac.yml
# tuning RocksDB with TPE tuner
nnictl create --config ./config_tpe.yml
```
