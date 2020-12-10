Hyper Parameter Optimization Comparison
=======================================

*Posted by Anonymous Author*

Comparison of Hyperparameter Optimization (HPO) algorithms on several problems.

Hyperparameter Optimization algorithms are list below:


* `Random Search <../Tuner/BuiltinTuner.rst>`__
* `Grid Search <../Tuner/BuiltinTuner.rst>`__
* `Evolution <../Tuner/BuiltinTuner.rst>`__
* `Anneal <../Tuner/BuiltinTuner.rst>`__
* `Metis <../Tuner/BuiltinTuner.rst>`__
* `TPE <../Tuner/BuiltinTuner.rst>`__
* `SMAC <../Tuner/BuiltinTuner.rst>`__
* `HyperBand <../Tuner/BuiltinTuner.rst>`__
* `BOHB <../Tuner/BuiltinTuner.rst>`__

All algorithms run in NNI local environment.

Machine Environment：

.. code-block:: bash

   OS: Linux Ubuntu 16.04 LTS
   CPU: Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz 2600 MHz
   Memory: 112 GB
   NNI Version: v0.7
   NNI Mode(local|pai|remote): local
   Python version: 3.6
   Is conda or virtualenv used?: Conda
   is running in docker?: no

AutoGBDT Example
----------------

Problem Description
^^^^^^^^^^^^^^^^^^^

Nonconvex problem on the hyper-parameter search of `AutoGBDT <../TrialExample/GbdtExample.rst>`__ example.

Search Space
^^^^^^^^^^^^

.. code-block:: json

   {
     "num_leaves": {
       "_type": "choice",
       "_value": [10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 48, 64, 96, 128]
     },
     "learning_rate": {
       "_type": "choice",
       "_value": [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
     },
     "max_depth": {
       "_type": "choice",
       "_value": [-1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 48, 64, 96, 128]
     },
     "feature_fraction": {
       "_type": "choice",
       "_value": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
     },
     "bagging_fraction": {
       "_type": "choice",
       "_value": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
     },
     "bagging_freq": {
       "_type": "choice",
       "_value": [1, 2, 4, 8, 10, 12, 14, 16]
     }
   }

The total search space is 1,204,224, we set the number of maximum trial to 1000. The time limitation is 48 hours.

Results
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Algorithm
     - Best loss
     - Average of Best 5 Losses
     - Average of Best 10 Losses
   * - Random Search
     - 0.418854
     - 0.420352
     - 0.421553
   * - Random Search
     - 0.417364
     - 0.420024
     - 0.420997
   * - Random Search
     - 0.417861
     - 0.419744
     - 0.420642
   * - Grid Search
     - 0.498166
     - 0.498166
     - 0.498166
   * - Evolution
     - 0.409887
     - 0.409887
     - 0.409887
   * - Evolution
     - 0.413620
     - 0.413875
     - 0.414067
   * - Evolution
     - 0.409887
     - 0.409887
     - 0.409887
   * - Anneal
     - 0.414877
     - 0.417289
     - 0.418281
   * - Anneal
     - 0.409887
     - 0.409887
     - 0.410118
   * - Anneal
     - 0.413683
     - 0.416949
     - 0.417537
   * - Metis
     - 0.416273
     - 0.420411
     - 0.422380
   * - Metis
     - 0.420262
     - 0.423175
     - 0.424816
   * - Metis
     - 0.421027
     - 0.424172
     - 0.425714
   * - TPE
     - 0.414478
     - 0.414478
     - 0.414478
   * - TPE
     - 0.415077
     - 0.417986
     - 0.418797
   * - TPE
     - 0.415077
     - 0.417009
     - 0.418053
   * - SMAC
     - **0.408386**
     - **0.408386**
     - **0.408386**
   * - SMAC
     - 0.414012
     - 0.414012
     - 0.414012
   * - SMAC
     - **0.408386**
     - **0.408386**
     - **0.408386**
   * - BOHB
     - 0.410464
     - 0.415319
     - 0.417755
   * - BOHB
     - 0.418995
     - 0.420268
     - 0.422604
   * - BOHB
     - 0.415149
     - 0.418072
     - 0.418932
   * - HyperBand
     - 0.414065
     - 0.415222
     - 0.417628
   * - HyperBand
     - 0.416807
     - 0.417549
     - 0.418828
   * - HyperBand
     - 0.415550
     - 0.415977
     - 0.417186
   * - GP
     - 0.414353
     - 0.418563
     - 0.420263
   * - GP
     - 0.414395
     - 0.418006
     - 0.420431
   * - GP
     - 0.412943
     - 0.416566
     - 0.418443


In this example, all the algorithms are used with default parameters. For Metis, there are about 300 trials because it runs slowly due to its high time complexity O(n^3) in Gaussian Process.

RocksDB Benchmark 'fillrandom' and 'readrandom'
-----------------------------------------------

Problem Description
^^^^^^^^^^^^^^^^^^^

`DB_Bench <https://github.com/facebook/rocksdb/wiki/Benchmarking-tools>`__ is the main tool that is used to benchmark `RocksDB <https://rocksdb.org/>`__\ 's performance. It has so many hapermeter to tune.

The performance of ``DB_Bench`` is associated with the machine configuration and installation method. We run the ``DB_Bench``\ in the Linux machine and install the Rock in shared library.

Machine configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   RocksDB:    version 6.1
   CPU:        6 * Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
   CPUCache:   35840 KB
   Keys:       16 bytes each
   Values:     100 bytes each (50 bytes after compression)
   Entries:    1000000

Storage performance
^^^^^^^^^^^^^^^^^^^

**Latency**\ : each IO request will take some time to complete, this is called the average latency. There are several factors that would affect this time including network connection quality and hard disk IO performance.

**IOPS**\ :** IO operations per second**\ , which means the amount of *read or write operations* that could be done in one seconds time.

**IO size**\ :** the size of each IO request**. Depending on the operating system and the application/service that needs disk access it will issue a request to read or write a certain amount of data at the same time.

**Throughput (in MB/s) = Average IO size x IOPS** 

IOPS is related to online processing ability and we use the IOPS as the metric in my experiment.

Search Space
^^^^^^^^^^^^

.. code-block:: json

   {
     "max_background_compactions": {
       "_type": "quniform",
       "_value": [1, 256, 1]
     },
     "block_size": {
       "_type": "quniform",
       "_value": [1, 500000, 1]
     },
     "write_buffer_size": {
       "_type": "quniform",
       "_value": [1, 130000000, 1]
     },
     "max_write_buffer_number": {
       "_type": "quniform",
       "_value": [1, 128, 1]
     },
     "min_write_buffer_number_to_merge": {
       "_type": "quniform",
       "_value": [1, 32, 1]
     },
     "level0_file_num_compaction_trigger": {
       "_type": "quniform",
       "_value": [1, 256, 1]
     },
     "level0_slowdown_writes_trigger": {
       "_type": "quniform",
       "_value": [1, 1024, 1]
     },
     "level0_stop_writes_trigger": {
       "_type": "quniform",
       "_value": [1, 1024, 1]
     },
     "cache_size": {
       "_type": "quniform",
       "_value": [1, 30000000, 1]
     },
     "compaction_readahead_size": {
       "_type": "quniform",
       "_value": [1, 30000000, 1]
     },
     "new_table_reader_for_compaction_inputs": {
       "_type": "randint",
       "_value": [1]
     }
   }

The search space is enormous (about 10^40) and we set the maximum number of trial to 100 to limit the computation resource.

Results
^^^^^^^

fillrandom' Benchmark
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Model
     - Best IOPS (Repeat 1)
     - Best IOPS (Repeat 2)
     - Best IOPS (Repeat 3)
   * - Random
     - 449901
     - 427620
     - 477174
   * - Anneal
     - 461896
     - 467150
     - 437528
   * - Evolution
     - 436755
     - 389956
     - 389790
   * - TPE
     - 378346
     - 482316
     - 468989
   * - SMAC
     - 491067
     - 490472
     - **491136**
   * - Metis
     - 444920
     - 457060
     - 454438


Figure:


.. image:: ../../img/hpo_rocksdb_fillrandom.png
   :target: ../../img/hpo_rocksdb_fillrandom.png
   :alt: 


'readrandom' Benchmark
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Model
     - Best IOPS (Repeat 1)
     - Best IOPS (Repeat 2)
     - Best IOPS (Repeat 3)
   * - Random
     - 2276157
     - 2285301
     - 2275142
   * - Anneal
     - 2286330
     - 2282229
     - 2284012
   * - Evolution
     - 2286524
     - 2283673
     - 2283558
   * - TPE
     - 2287366
     - 2282865
     - 2281891
   * - SMAC
     - 2270874
     - 2284904
     - 2282266
   * - Metis
     - **2287696**
     - 2283496
     - 2277701


Figure:


.. image:: ../../img/hpo_rocksdb_readrandom.png
   :target: ../../img/hpo_rocksdb_readrandom.png
   :alt: 

