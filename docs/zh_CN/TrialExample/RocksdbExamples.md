# 在 NNI 上调优 RocksDB

## 概述

[RocksDB](https://github.com/facebook/rocksdb) 是流行的高性能、嵌入式的生产级别的键值数据库，它在 Facebook，Yahoo!，和 LinkedIn 等各种规模的网站上使用。 这是 Facebook 的 [LevelDB](https://github.com/google/leveldb) 分支，为多 CPU，快速存储（如 SSD）的输入输出进行了优化。

RocksDB 的性能表现非常依赖于调优操作。 但由于其底层技术较复杂，可配置参数非常多，很难获得较好的配置。 NNI 可帮助解决此问题。 NNI 支持多种调优算法来为 RocksDB 搜索最好的配置，并支持本机、远程服务器和云服务等多种环境。

本示例展示了如何使用 NNI，通过评测工具 `db_bench` 来找到 `fillrandom` 基准的最佳配置，此工具是 RocksDB 官方提供的评测工具。 在运行示例前，需要检查 NNI 已安装，

db_bench</code> 已经加入到了 `PATH` 中。 Please refer to [here](../Tutorial/QuickStart.md) for detailed information about installation and preparing of NNI environment, and [here](https://github.com/facebook/rocksdb/blob/master/INSTALL.md) for compiling RocksDB as well as `db_bench`.</p> 

We also provide a simple script [`db_bench_installation.sh`](../../../examples/trials/systems/rocksdb-fillrandom/db_bench_installation.sh) helping to compile and install `db_bench` as well as its dependencies on Ubuntu. Installing RocksDB on other systems can follow the same procedure.

*code directory: [`example/trials/systems/rocksdb-fillrandom`](../../../examples/trials/systems/rocksdb-fillrandom)*



## Experiment setup

There are mainly three steps to setup an experiment of tuning systems on NNI. Define search space with a `json` file, write a benchmark code, and start NNI experiment by passing a config file to NNI manager.



### Search Space

For simplicity, this example tunes three parameters, `write_buffer_size`, `min_write_buffer_num` and `level0_file_num_compaction_trigger`, for writing 16M keys with 20 Bytes of key size and 100 Bytes of value size randomly, based on writing operations per second (OPS). `write_buffer_size` sets the size of a single memtable. Once memtable exceeds this size, it is marked immutable and a new one is created. `min_write_buffer_num` is the minimum number of memtables to be merged before flushing to storage. Once the number of files in level 0 reaches `level0_file_num_compaction_trigger`, level 0 to level 1 compaction is triggered.

In this example, the search space is specified by a `search_space.json` file as shown below. Detailed explanation of search space could be found [here](../Tutorial/SearchSpaceSpec.md).



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


*code directory: [`example/trials/systems/rocksdb-fillrandom/search_space.json`](../../../examples/trials/systems/rocksdb-fillrandom/search_space.json)*



### Benchmark code

Benchmark code should receive a configuration from NNI manager, and report the corresponding benchmark result back. Following NNI APIs are designed for this purpose. In this example, writing operations per second (OPS) is used as a performance metric. Please refer to [here](Trials.md) for detailed information.

* Use `nni.get_next_parameter()` to get next system configuration.
* Use `nni.report_final_result(metric)` to report the benchmark result.

*code directory: [`example/trials/systems/rocksdb-fillrandom/main.py`](../../../examples/trials/systems/rocksdb-fillrandom/main.py)*



### Config file

One could start a NNI experiment with a config file. A config file for NNI is a `yaml` file usually including experiment settings (`trialConcurrency`, `maxExecDuration`, `maxTrialNum`, `trial gpuNum`, etc.), platform settings (`trainingServicePlatform`, etc.), path settings (`searchSpacePath`, `trial codeDir`, etc.) and tuner settings (`tuner`, `tuner optimize_mode`, etc.). Please refer to [here](../Tutorial/QuickStart.md) for more information.

Here is an example of tuning RocksDB with SMAC algorithm:

*code directory: [`example/trials/systems/rocksdb-fillrandom/config_smac.yml`](../../../examples/trials/systems/rocksdb-fillrandom/config_smac.yml)*

Here is an example of tuning RocksDB with TPE algorithm:

*code directory: [`example/trials/systems/rocksdb-fillrandom/config_tpe.yml`](../../../examples/trials/systems/rocksdb-fillrandom/config_tpe.yml)*

Other tuners can be easily adopted in the same way. Please refer to [here](../Tuner/BuiltinTuner.md) for more information.

Finally, we could enter the example folder and start the experiment using following commands:



```bash
# tuning RocksDB with SMAC tuner
nnictl create --config ./config_smac.yml
# tuning RocksDB with TPE tuner
nnictl create --config ./config_tpe.yml
```




## Experiment results

We ran these two examples on the same machine with following details:

* 16 * Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz
* 465 GB of rotational hard drive with ext4 file system
* 128 GB of RAM
* Kernel version: 4.15.0-58-generic
* NNI version: v1.0-37-g1bd24577
* RocksDB version: 6.4
* RocksDB DEBUG_LEVEL: 0

The detailed experiment results are shown in the below figure. Horizontal axis is sequential order of trials. Vertical axis is the metric, write OPS in this example. Blue dots represent trials for tuning RocksDB with SMAC tuner, and orange dots stand for trials for tuning RocksDB with TPE tuner. 

![image](../../../examples/trials/systems/rocksdb-fillrandom/plot.png)

Following table lists the best trials and corresponding parameters and metric obtained by the two tuners. Unsurprisingly, both of them found the same optimal configuration for `fillrandom` benchmark.

| Tuner | Best trial | Best OPS | write_buffer_size | min_write_buffer_number_to_merge | level0_file_num_compaction_trigger |
|:-----:|:----------:|:--------:|:-------------------:|:------------------------------------:|:--------------------------------------:|
| SMAC  |    255     |  779289  |       2097152       |                 7.0                  |                  7.0                   |
|  TPE  |    169     |  761456  |       2097152       |                 7.0                  |                  7.0                   |
