# 使用 NNI 调优 RocksDB

## 概述

[RocksDB](https://github.com/facebook/rocksdb) 是一种很受欢迎的高性能嵌入式键值数据库，被许多公司，如 Facebook， Yahoo! 和 LinkedIn 等，广泛应用于各种网络规模的产品中。它是 Facebook 在 [LevelDB](https://github.com/google/leveldb) 的基础上，通过充分利用多核心中央处理器和快速存储器（如固态硬盘）的特点，针对IO密集型应用优化而成的。

RocksDB 的性能高度依赖于运行参数的调优。然而，由于其底层技术极为复杂，需要调整的参数过多，有时很难找到合适的运行参数。NNI 可以帮助数据库运维工程师解决这个问题。NNI 支持多种自动调参算法，并且支持运行于本地、远程和云端的各种负载。

本例展示了如何使用 NNI 搜索 RocksDB 在 `fillrandom` 基准测试中的最佳运行参数，`fillrandom` 基准测试是 RocksDB 官方提供的基准测试工具 `db_bench` 所支持的一种基准测试，因此在运行本例之前请确保您已经安装了 NNI，并且 `db_bench` 在您的 `PATH` 路径中。关于如何安装和准备 NNI 环境，请参考[这里](../Tuner/BuiltinTuner.md)，关于如何编译 RocksDB 以及 `db_bench`，请参考[这里](https://github.com/facebook/rocksdb/blob/master/INSTALL.md)。

我们还提供了一个简单的脚本 [`db_bench_installation.sh`](../../../examples/trials/systems/rocksdb-fillrandom/db_bench_installation.sh)，用来在 Ubuntu 系统上编译和安装 `db_bench` 和相关依赖。在其他系统中的安装也可以参考该脚本实现。

*代码目录: [`example/trials/systems/rocksdb-fillrandom`](../../../examples/trials/systems/rocksdb-fillrandom)*

## 实验配置

使用 NNI 进行调优系统主要有三个步骤，分别是，使用一个 `json` 文件定义搜索空间；准备一个基准测试程序；和一个用来启动 NNI 实验的配置文件。

### 搜索空间

简便起见，本例基于 Rocks_DB 每秒的写入操作数（Operations Per Second, OPS），在随机写入 16M 个键长为 20 字节值长为 100 字节的键值对的情况下，对三个系统运行参数，`write_buffer_size`，`min_write_buffer_num` 和 `level0_file_num_compaction_trigger`，进行了调优。`write_buffer_size` 控制了单个 memtable 的大小。在写入过程中，当 memtable 的大小超过了 `write_buffer_size` 指定的数值，该 memtable 将会被标记为不可变，并创建一个新的 memtable。`min_write_buffer_num` 是在写入（flush）磁盘之前需要合并（merge）的 memtable 的最小数量。一旦 level 0 中的文件数量超过了 `level0_file_num_compaction_trigger` 所指定的数，level 0 向 level 1 的压缩（compaction）将会被触发。

搜索空间由如下所示的文件 `search_space.json` 指定。更多关于搜索空间的解释请参考[这里](../Tutorial/SearchSpaceSpec.md)。

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

*代码目录: [`example/trials/systems/rocksdb-fillrandom/search_space.json`](../../../examples/trials/systems/rocksdb-fillrandom/search_space.json)*

### 基准测试

基准测试程序需要从 NNI manager 接收一个运行参数，并在运行基准测试以后向 NNI manager 汇报基准测试结果。NNI 提供了下面两个 APIs 来完成这些任务。更多关于 NNI trials 的信息请参考[这里](Trials.md)。

* 使用 `nni.get_next_parameter()` 从 NNI manager 得到需要测试的系统运行参数。
* 使用 `nni.report_final_result(metric)` 向 NNI manager 汇报基准测试的结果。

*代码目录: [`example/trials/systems/rocksdb-fillrandom/main.py`](../../../examples/trials/systems/rocksdb-fillrandom/main.py)*

### 配置文件

NNI 实验可以通过配置文件来启动。通常而言，NNI 配置文件是一个 `yaml` 文件，通常包含实验设置（`trialConcurrency`，`maxExecDuration`，`maxTrialNum`，`trial gpuNum` 等），运行平台设置（`trainingServicePlatform` 等），路径设置（`searchSpacePath`，`trial codeDir` 等）和 调参器设置（`tuner`，`tuner optimize_mode` 等）。更多关于 NNI 配置文件的信息请参考[这里](../Tutorial/QuickStart.md)。

下面是使用 SMAC 算法调优 RocksDB 配置文件的例子：

*代码目录: [`example/trials/systems/rocksdb-fillrandom/config_smac.yml`](../../../examples/trials/systems/rocksdb-fillrandom/config_smac.yml)*

下面是使用 TPE 算法调优 RocksDB 配置文件的例子：

*代码目录: [`example/trials/systems/rocksdb-fillrandom/config_tpe.yml`](../../../examples/trials/systems/rocksdb-fillrandom/config_tpe.yml)*

其他的调参器可以使用同样的方式应用，更多关于调参器的信息请参考[这里](../Tuner/BuiltinTuner.md)。

最后，我们可以进入本例的文件夹内，用下面的命令启动实验：

```bash
# tuning RocksDB with SMAC tuner
nnictl create --config ./config_smac.yml
# tuning RocksDB with TPE tuner
nnictl create --config ./config_tpe.yml
```

## 实验结果

我们在同一台机器上运行了这两个实验，相关信息如下：

* 16 * Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz
* 465 GB of rotational hard drive with ext4 file system
* 128 GB of RAM
* Kernel version: 4.15.0-58-generic
* NNI version: v1.0-37-g1bd24577
* RocksDB version: 6.4
* RocksDB DEBUG_LEVEL: 0

具体的实验结果如下图所示。横轴是基准测试的顺序，纵轴是基准测试得到的结果，在本例中是每秒钟写操作的次数。蓝色的圆点代表用 SMAC 调优 RocksDB 得到的基准测试结果，而橘黄色的圆点表示用 TPE 调优得到的基准测试结果。

![image](../../../examples/trials/systems/rocksdb-fillrandom/plot.png)

下面的表格列出了使用两种调参器得到的最好的基准测试结果及相对应的参数。毫不意外，使用这两种调参器在 `fillrandom` 基准测试中搜索得到了相同的最优参数。

| Tuner | Best trial | Best OPS | write_buffer_size | min_write_buffer_number_to_merge | level0_file_num_compaction_trigger |
| :---: | :--------: | :------: | :---------------: | :------------------------------: | :--------------------------------: |
| SMAC  | 255        | 779289   | 2097152           | 7.0                              | 7.0                                |
| TPE   | 169        | 761456   | 2097152           | 7.0                              | 7.0                                |
