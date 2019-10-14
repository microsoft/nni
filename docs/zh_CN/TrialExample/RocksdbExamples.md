# 使用 NNI 调优 RocksDB

## 概述

[RocksDB](https://github.com/facebook/rocksdb) 是一种很受欢迎的高性能嵌入式键值数据库，被许多公司，如 Facebook， Yahoo! 和 LinkedIn 等，广泛应用于各种网络规模的产品中。它是 Facebook 在 [LevelDB](https://github.com/google/leveldb) 的基础上，通过充分利用多核心中央处理器和快速存储的特点，针对IO密集型应用优化而成的。

RocksDB 的性能高度依赖于运行参数的调优。然而，由于其底层技术极为复杂，需要调整的参数过多，有时很难找到合适的运行参数。NNI 可以帮助数据库运维工程师解决这个问题。NNI 支持多种自动调参算法，并且支持运行于本地、远程和云端的各种负载。本例展示了如何使用 NNI 中的 SMAC 和 TPE 调参器搜索 RocksDB 的最佳运行参数，使其在随机写实验中获得最好的性能。其他调参器的用法也很类似。更详细的信息可以在[这里](../Tuner/BuiltinTuner.md)找到。

`代码目录: example/trials/systems/rocksdb-fillrandom`

## 目标

本例将展示如何使用 NNI 来搜索 RocksDB 在 `fillrandom` 基准测试中性能最好的运行参数。`fillrandom` 基准测试是 RocksDB 官方提供的基准测试工具 `db_bench` 所支持的一种基准测试，因此在运行本例之前请确保您已经安装了 NNI，并且 `db_bench` 在您的 `PATH` 路径中。关于如何安装和准备 NNI 环境，请参考[这里](../Tuner/BuiltinTuner.md)，关于如何编译 RocksDB 以及 `db_bench`，请参考[这里](https://github.com/facebook/rocksdb/blob/master/INSTALL.md)。

## 搜索空间

简便起见，本例基于 Rocks_DB 每秒的写入操作数（Operations Per Second, OPS），在随机写入 16M 个键长为 20 字节值长为 100 字节的键值对的情况下，对三个系统运行参数，`buffer_size`，`min_write_buffer_num` 和 `level0_file_num_compaction_trigger`，进行了调优。搜索空间由如下所示的文件 `search_space.json` 指定。更多关于搜索空间的解释请参考[这里](https://github.com/microsoft/nni/blob/master/docs/en_US/Tutorial/SearchSpaceSpec.md)。

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

`代码目录: example/trials/systems/rocksdb-fillrandom/search_space.json`

## 基准测试

基准测试程序需要从 NNI manager 接收一个运行参数，并在运行基准测试以后向 NNI manager 汇报基准测试结果。NNI 提供了下面两个 APIs 来完成这些任务。更多关于 NNI trials 的信息请参考[这里](Trials.md)。

* 使用 `nni.get_next_parameter()` 从 NNI manager 得到需要测试的系统运行参数。
* 使用 `nni.report_final_result(metric)` 向 NNI manager 汇报基准测试的结果。

`代码目录: example/trials/systems/rocksdb-fillrandom/main.py`

## NNI 配置文件

NNI 实验可以通过配置文件来启动。通常而言，NNI 配置文件需要包括实验设置（`trialConcurrency`，`maxExecDuration`，`maxTrialNum`，`trial gpuNum` 等），运行平台设置（`trainingServicePlatform` 等），路径设置（`searchSpacePath`，`trial codeDir` 等）和 调参器设置（`tuner`，`tuner optimize_mode` 等）。更多关于 NNI 配置文件的信息请参考[这里](../Tutorial/QuickStart.md)。

下面是使用 SMAC 算法调优 RocksDB 配置文件的例子：

`代码目录: examples/trials/systems/rocksdb-fillrandom/config_smac.yml`

下面是使用 TPE 算法调优 RocksDB 配置文件的例子：

`代码目录: examples/trials/systems/rocksdb-fillrandom/config_tpe.yml`

## 运行调优实验

以上文件即为本例包含的主要内容。进入本例文件夹内，用下面的命令即可启动实验：

```bash
# tuning RocksDB with SMAC tuner
nnictl create --config ./config_smac.yml
# tuning RocksDB with TPE tuner
nnictl create --config ./config_tpe.yml
```