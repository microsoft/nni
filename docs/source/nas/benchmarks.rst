NAS Benchmark
=============

.. note:: :doc:`Example usage of NAS benchmarks </tutorials/nasbench_as_dataset>`.

To improve the reproducibility of NAS algorithms as well as reducing computing resource requirements, researchers proposed a series of NAS benchmarks such as `NAS-Bench-101 <https://arxiv.org/abs/1902.09635>`__, `NAS-Bench-201 <https://arxiv.org/abs/2001.00326>`__, `NDS <https://arxiv.org/abs/1905.13214>`__, etc. NNI provides a query interface for users to acquire these benchmarks. Within just a few lines of code, researcher are able to evaluate their NAS algorithms easily and fairly by utilizing these benchmarks.

Prerequisites
-------------

* Please prepare a folder to household all the benchmark databases. By default, it can be found at ``${HOME}/.cache/nni/nasbenchmark``. Or you can place it anywhere you like, and specify it in ``NASBENCHMARK_DIR`` via ``export NASBENCHMARK_DIR=/path/to/your/nasbenchmark`` before importing NNI.
* Please install ``peewee`` via ``pip3 install peewee``, which NNI uses to connect to database.

Data Preparation
----------------

You can download the preprocessed benchmark files via ``python -m nni.nas.benchmark.download <benchmark_name>``, where ``<benchmark_name>`` can be ``nasbench101``, ``nasbench201``, and etc. Add ``--help`` to the command for supported command line arguments.

Example Usages
--------------

Please refer to :githublink:`test/algo/nas/benchmark/test_algo.py` on how to test algorithms on NAS benchmarks.

Supported benchmarks
--------------------

NAS-Bench-101
^^^^^^^^^^^^^

* `Paper link <https://arxiv.org/abs/1902.09635>`__ 
* `Open-source <https://github.com/google-research/nasbench>`__

NAS-Bench-101 contains 423,624 unique neural networks, combined with 4 variations in number of epochs (4, 12, 36, 108), each of which is trained 3 times. It is a cell-wise search space, which constructs and stacks a cell by enumerating DAGs with at most 7 operators, and no more than 9 connections. All operators can be chosen from ``CONV3X3_BN_RELU``, ``CONV1X1_BN_RELU`` and ``MAXPOOL3X3``, except the first operator (always ``INPUT``\ ) and last operator (always ``OUTPUT``\ ).

Notably, NAS-Bench-101 eliminates invalid cells (e.g., there is no path from input to output, or there is redundant computation). Furthermore, isomorphic cells are de-duplicated, i.e., all the remaining cells are computationally unique.

NAS-Bench-201
^^^^^^^^^^^^^

* `Paper link <https://arxiv.org/abs/2001.00326>`__ 
* `Open-source API <https://github.com/D-X-Y/NAS-Bench-201>`__ 
* `Implementations <https://github.com/D-X-Y/AutoDL-Projects>`__

NAS-Bench-201 is a cell-wise search space that views nodes as tensors and edges as operators. The search space contains all possible densely-connected DAGs with 4 nodes, resulting in 15,625 candidates in total. Each operator (i.e., edge) is selected from a pre-defined operator set (\ ``NONE``, ``SKIP_CONNECT``, ``CONV_1X1``, ``CONV_3X3`` and ``AVG_POOL_3X3``\ ). Training appraoches vary in the dataset used (CIFAR-10, CIFAR-100, ImageNet) and number of epochs scheduled (12 and 200). Each combination of architecture and training approach is repeated 1 - 3 times with different random seeds.

NDS
^^^

* `Paper link <https://arxiv.org/abs/1905.13214>`__ 
* `Open-source <https://github.com/facebookresearch/nds>`__

*On Network Design Spaces for Visual Recognition* released trial statistics of over 100,000 configurations (models + hyper-parameters) sampled from multiple model families, including vanilla (feedforward network loosely inspired by VGG), ResNet and ResNeXt (residual basic block and residual bottleneck block) and NAS cells (following popular design from NASNet, Ameoba, PNAS, ENAS and DARTS). Most configurations are trained only once with a fixed seed, except a few that are trained twice or three times.

Instead of storing results obtained with different configurations in separate files, we dump them into one single database to enable comparison in multiple dimensions. Specifically, we use ``model_family`` to distinguish model types, ``model_spec`` for all hyper-parameters needed to build this model, ``cell_spec`` for detailed information on operators and connections if it is a NAS cell, ``generator`` to denote the sampling policy through which this configuration is generated. Refer to API documentation for details.

Here is a list of available operators used in NDS.

.. automodule:: nni.nas.benchmark.nds.constants
   :noindex:

.. warning::

   NDS benchmark doesn't support benchmarking with existing algorithms because the authors only released a subset of data points in the search space. Benchmarking on NDS is a work in progress.
