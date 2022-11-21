Use Cases and Solutions
=======================

Different from the tutorials and examples in the rest of the document which show the usage of a feature, this part mainly introduces end-to-end scenarios and use cases to help users further understand how NNI can help them. NNI can be widely adopted in various scenarios. We also encourage community contributors to share their AutoML practices especially the NNI usage practices from their experience.

Automatic Model Tuning
----------------------

NNI can be applied on various model tuning tasks. Some state-of-the-art model search algorithms, such as EfficientNet, can be easily built on NNI. Popular models, e.g., recommendation models, can be tuned with NNI. The following are some use cases to illustrate how to leverage NNI in your model tuning tasks and how to build your own pipeline with NNI.

* :doc:`Tuning SVD automatically <recommenders_svd>`
* :doc:`EfficientNet on NNI <efficientnet>`
* :doc:`Automatic Model Architecture Search for Reading Comprehension <squad_evolution_examples>`
* :doc:`Parallelizing Optimization for TPE <parallelizing_tpe_search>`

Automatic System Tuning
-----------------------

The performance of systems, such as database, tensor operator implementaion, often need to be tuned to adapt to specific hardware configuration, targeted workload, etc. Manually tuning a system is complicated and often requires detailed understanding of hardware and workload. NNI can make such tasks much easier and help system owners find the best configuration to the system automatically. The detailed design philosophy of automatic system tuning can be found in this `paper <https://dl.acm.org/doi/10.1145/3352020.3352031>`__ . The following are some typical cases that NNI can help.

* :doc:`Tuning SPTAG (Space Partition Tree And Graph) automatically <sptag_auto_tune>`
* :doc:`Tuning the performance of RocksDB <rocksdb_examples>`
* :doc:`Tuning Tensor Operators automatically <op_evo_examples>`

Feature Engineering
-------------------

The following is an article about how NNI helps in auto feature engineering shared by a community contributor. More use cases and solutions will be added in the future.

* :doc:`NNI review article from Zhihu: - By Garvin Li <nni_autofeatureeng>`

Performance Measurement, Comparison and Analysis
------------------------------------------------

Performance comparison and analysis can help users decide a proper algorithm (e.g., tuner, NAS algorithm) for their scenario. The following are some measurement and comparison data for users' reference.

* :doc:`Neural Architecture Search Comparison <nas_comparison>`
* :doc:`Hyper-parameter Tuning Algorithm Comparsion <hpo_comparison>`
