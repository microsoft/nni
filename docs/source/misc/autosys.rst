Automatic System Tuning
=======================

The performance of systems, such as database, tensor operator implementaion, often need to be tuned to adapt to specific hardware configuration, targeted workload, etc. Manually tuning a system is complicated and often requires detailed understanding of hardware and workload. NNI can make such tasks much easier and help system owners find the best configuration to the system automatically. The detailed design philosophy of automatic system tuning can be found in this `paper <https://dl.acm.org/doi/10.1145/3352020.3352031>`__\ . The following are some typical cases that NNI can help.

..  toctree::
    :maxdepth: 1

    Tuning SPTAG (Space Partition Tree And Graph) automatically <sptag_auto_tune>
    Tuning the performance of RocksDB <rocksdb_examples>
    Tuning Tensor Operators automatically <op_evo_examples>