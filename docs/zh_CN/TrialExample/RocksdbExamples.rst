在 NNI 上调优 RocksDB
=====================

概述
--------

`RocksDB <https://github.com/facebook/rocksdb>`__ 是流行的高性能、嵌入式的生产级别的键值数据库，它在 Facebook，Yahoo!，和 LinkedIn 等各种规模的网站上使用。 这是 Facebook 的 `LevelDB <https://github.com/google/leveldb>`__ 分支，为多 CPU，快速存储（如 SSD）的输入输出进行了优化。

RocksDB 的性能表现非常依赖于调优操作。 但由于其底层技术较复杂，可配置参数非常多，很难获得较好的配置。 NNI 可帮助解决此问题。 NNI 支持多种调优算法来为 RocksDB 搜索最好的配置，并支持本机、远程服务器和云服务等多种环境。 

本示例展示了如何使用 NNI，通过评测工具 ``db_bench`` 来找到 ``fillrandom`` 基准的最佳配置，此工具是 RocksDB 官方提供的评测工具。 在运行示例前，需要检查 NNI 已安装， `db_bench <https://github.com/facebook/rocksdb/wiki/Benchmarking-tools>`__ 已经加入到了 ``PATH`` 中。 参考 `这里 <../Tutorial/QuickStart.rst>`__ ，了解如何安装并准备 NNI 环境，参考 `这里 <https://github.com/facebook/rocksdb/blob/master/INSTALL.rst>`__ 来编译 RocksDB 以及 ``db_bench``。

此简单脚本 :githublink:`db_bench_installation.sh <examples/trials/systems/rocksdb-fillrandom/db_bench_installation.sh>` 可帮助编译并在 Ubuntu 上安装 ``db_bench`` 及其依赖包。 可遵循相同的过程在其它系统中安装 RocksDB。

代码目录： :githublink:`example/trials/systems/rocksdb-fillrandom <examples/trials/systems/rocksdb-fillrandom>`

Experiment 设置
----------------

在 NNI 上配置调优的 Experiment 主要有三个步骤。 使用 ``json`` 文件定义搜索空间，编写评测代码，将配置传入 NNI 管理器来启动 Experiment。

搜索空间
^^^^^^^^^^^^

为简单起见，此示例仅调优三个参数，``write_buffer_size``，``min_write_buffer_num`` 以及 ``level0_file_num_compaction_trigger``，场景为测试随机写入 16M 数据的每秒写操作数（OPS），其 Key 为 20 字节，值为 100 字节。 ``write_buffer_size`` 设置单个内存表的大小。 一旦内存表超过此大小，会被标记为不可变，并创建新内存表。 ``min_write_buffer_num`` 是要合并写入存储的最小内存表数量。 一旦 Level 0 的文件数量达到了 ``level0_file_num_compaction_trigger``，就会触发 Level 0 到 Level 1 的压缩。

此示例中，下列 ``search_space.json`` 文件指定了搜索空间。 搜索空间的详细说明参考 `这里 <../Tutorial/SearchSpaceSpec.rst>`__.。

.. code-block:: json

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

代码目录 :githublink:`example/trials/systems/rocksdb-fillrandom/search_space.json <examples/trials/systems/rocksdb-fillrandom/search_space.json>`

基准测试
^^^^^^^^^^^^^^

评测代码从 NNI 管理器接收配置，并返回相应的基准测试结果。 下列 NNI API 用于相应的操作。 此示例中，每秒写操作数（OPS）作为了性能指标。 参考 `这里 <Trials.rst>`__，了解详情。


* 使用 ``nni.get_next_parameter()`` 来获取下一个系统配置。
* 使用 ``nni.report_final_result(metric)`` 来返回测试结果。

代码目录 :githublink:`example/trials/systems/rocksdb-fillrandom/main.py <examples/trials/systems/rocksdb-fillrandom/main.py>`

配置文件
^^^^^^^^^^^

用于启动 NNI Experiment 的配置文件。 此配置文件是 ``YAML`` 格式，通常包括了 Experiment 设置 (\ ``trialConcurrency``\ , ``maxExecDuration``\ , ``maxTrialNum``\ , ``trial gpuNum``\ 等)，平台设置 (\ ``trainingServicePlatform``\ 等)，路径设置 (\ ``searchSpacePath``\ , ``trial codeDir``\ 等) 以及 Tuner 设置 (\ ``tuner``\ , ``tuner optimize_mode``\ 等)。 参考 `这里 <../Tutorial/QuickStart.rst>`__ 了解详情。

这是使用 SMAC 算法调优 RocksDB 的示例：

代码目录 :githublink:`example/trials/systems/rocksdb-fillrandom/config_smac.yml <examples/trials/systems/rocksdb-fillrandom/config_smac.yml>`

这是使用 TPE 算法调优 RocksDB 的示例：

代码目录 :githublink:`example/trials/systems/rocksdb-fillrandom/config_tpe.yml <examples/trials/systems/rocksdb-fillrandom/config_tpe.yml>`

其它 Tuner 算法可以通过相同的方式来使用。 参考 `这里 <../Tuner/BuiltinTuner.rst>`__ 了解详情。

最后，进入示例目录，并通过下列命令启动 Experiment：

.. code-block:: bash

   # 在 NNI 上调优 RocksDB
   nnictl create --config ./config_smac.yml
   # 在 NNI 上使用 TPE Tuner 调优 RocksDB
   nnictl create --config ./config_tpe.yml

Experiment 结果
------------------

在同一台计算机上运行这两个示例的详细信息：


* 16 * Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz
* 465 GB 磁盘，安装 ext4 操作系统
* 128 GB 内存
* 内核版本: 4.15.0-58-generic
* NNI 版本: v1.0-37-g1bd24577
* RocksDB 版本: 6.4
* RocksDB DEBUG_LEVEL: 0

详细的实验结果如下图所示。 水平轴是 Trial 的顺序。 垂直轴是指标，此例中为写入的 OPS。 蓝点表示使用的是 SMAC Tuner，橙色表示使用的是 TPE Tuner。 


.. image:: https://github.com/microsoft/nni/blob/v2.0/docs/img/rocksdb-fillrandom-plot.png?raw=true


下表列出了两个 Tuner 获得的最佳 Trial 以及相应的参数和指标。 不出所料，两个 Tuner 都为 ``fillrandom`` 测试找到了一样的最佳配置。

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 概述
     - 最佳 Trial
     - 最佳 OPS
     - write_buffer_size
     - min_write_buffer_number_to_merge
     - level0_file_num_compaction_trigger
   * - SMAC
     - 255
     - 779289
     - 2097152
     - 7.0
     - 7.0
   * - TPE
     - 169
     - 761456
     - 2097152
     - 7.0
     - 7.0

