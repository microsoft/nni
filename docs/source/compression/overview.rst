Overview of NNI Model Compression
=================================

The NNI model compression has undergone a completely new framework design in version 3.0,
seamlessly integrating pruning, quantization, and distillation methods.
Additionally, it provides a more granular model compression configuration,
including compression granularity configuration, input/output compression configuration, and custom module compression.
Furthermore, the model speedup part of pruning uses the graph analysis scheme based on torch.fx,
which supports more op types of sparsity propagation,
as well as custom special op sparsity propagation methods and replacement logic,
further enhancing the generality and robustness of model acceleration.

The current documentation for the new version of compression may not be complete, but there is no need to worry.
The optimizations in the new version are mostly focused on the underlying framework and implementation,
and there are not significant changes to the user interface.
Instead, there are more extensions and compatibility with the configuration of the previous version.

If you want to view the old compression documents, please refer `nni 2.10 compression doc <https://nni.readthedocs.io/en/v2.10/compression/overview.html>`__.

See :doc:`the major enhancement of compression in NNI 3.0 <./changes>`.
