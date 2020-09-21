#######################
自动系统调优
#######################

数据库、张量算子实现等系统的性能往往需要进行调优，以适应特定的硬件配置、目标工作负载等。 手动调优系统非常复杂，并且通常需要对硬件和工作负载有详细的了解。 NNI 可以使这些任务变得更容易，并帮助系统所有者自动找到系统的最佳配置。 自动系统调优的详细设计思想可以在[这篇文章](https://dl.acm.org/doi/10.1145/3352020.3352031)中找到。 以下是 NNI 可以发挥作用的一些典型案例。

..  toctree::
    :maxdepth: 1

    自动调优 SPTAG（Space Partition Tree And Graph）<SptagAutoTune>
    调优 RocksDB 的性能<../TrialExample/RocksdbExamples>
    自动调优张量算子<../TrialExample/OpEvoExamples>