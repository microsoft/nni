Main Changes of Compression API
===============================

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

