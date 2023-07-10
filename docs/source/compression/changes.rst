Major Enhancement of Compression in NNI 3.0
===========================================

To bolster additional compression scenarios and more particular compression configurations,
we have revised the compression application programming interface (API) in NNI 3.0.
If you are a beginner to NNI Compression, you could bypass this document.
Nonetheless, if you have employed NNI Compression before and want to try the latest Compression version,
this document will help you in comprehending the noteworthy alterations in the interface in 3.0.


Compression Target
------------------

The notion of ``compression target`` is a novel concept introduced in NNI 3.0.
It refers to the specific parts of a module that should be compressed, such as input, output or weights.

In previous versions, NNI assumed that all module types should have parameters named ``weight`` and ``bias``,
and only produced masks for these parameters.
This assumption was suitable for a significant degree of simulation compression.
However, it is undeniable that there are still many modules that do not fit into this assumption,
particularly for customized modules.

Therefore, in NNI 3.0, model compression can configure specifically for the level of input, output, and parameters of the module.
By means of fine-grained configuration, NNI can not only compress module types that were previously uncompressible,
but also achieve better simulation compression.
As a result, the gap in accuracy between simulation compression and real speedup becomes extremely small.

For instance, in previous versions, the operation of ``softmax`` would significantly diminish the effect of simulated pruning,
since 0 as input is also meaningful for ``softmax``.
In NNI 3.0, this can be avoided by setting the input and output masks and ``apply_method``
to ensure that ``softmax`` obtains the correct simulated pruning result.

Please consult the sections on :ref:`target_names` and :ref:`target_settings` for further details.


Compression Mode
----------------

In the previous version of NNI (lower than 3.0), three pruning modes were supported: ``normal``, ``global``, and ``dependency-aware``.

In the ``normal`` mode, each module was required to be assigned a sparse ratio, and the pruner generated masks directly on the weight elements of this ratio.

In the ``global`` mode, a sparse ratio was set for a group of modules, and the pruner generated masks whose overall sparse ratio conformed to the setting,
but the sparsity of each module in the group may differ.

The ``dependency-aware`` mode constrained modules with operational dependencies to generate related masks.

For instance, if the outputs of two modules had an ``add`` relationship, then the two modules would have the same masks in the output dimension.

Different modes were better suited to different compression scenarios to achieve improved compression effects.
Nevertheless, we believe that more flexible combinations should be allowed.
For example, in a compression process, certain modules of similar levels could apply the overall sparse ratio,
while other modules with operational dependencies could generate similar masks at the same time.

Right now in NNI 3.0, users can directly set :ref:`global_group_id` and :ref:`dependency_group_id` to implement ``global`` and ``dependency-aware`` modes.
Additionally, :ref:`align` is supported to generate a mask from another module mask, such as generating a batch normalization mask from a convolution mask.
You can achieve improved performance and exploration by combining these modes by setting the appropriate keys in the configuration list.


Pruning Speedup
---------------

The previous method of pruning speedup relied on ``torch.jit.trace`` to trace the model graph.
However, this method had several limitations and required additional support to perform certain operations.
These limitations resulted in excessive maintenance costs, making it difficult to continue development. 

To address these issues, in NNI 3.0, we refactored the pruning speedup based on ``concrete_trace``.
This is a useful utility for tracing a model graph, based on ``torch.fx``.
Unlike ``torch.fx.symbolic_trace``, ``concrete_trace`` executes the entire model, resulting in a more complete graph.
As a result, most operations that couldn't be traced in the previous pruning speedup can now be traced. 

In addition to ``concrete_trace``, users who have a good ``torch.fx.GraphModule`` for their traced model can also use the ``torch.fx.GraphModule`` directly.
Furthermore, the new pruning speedup supports customized masks propagation logic and module replacement methods to cope with the speedup of various customized modules.

Model Fusion
------------

Model fusion is supported in NNI 3.0. You can use it easily by setting ``fuse_names`` in each configure in the config_list.
Please refer :doc:`Module Fusion <./module_fusion>` for more details.

Distillation
------------

Two distillers is supported in NNI 3.0. By pruning or quantization fused distillation, it can get better compression results and higher precision.

Please refer :doc:`Distiller <../reference/compression/distiller>` for more details.


Fusion Compression
------------------

Thanks to the new unified compression framework, it is now possible to perform pruning, quantization, and distillation simultaneously,
without having to apply them one by one.

Please refer :doc:`fusion compression <./fusion_compress>` for more details.
