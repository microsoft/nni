Fusion Compression
==================

Fusion compression is a novel experimental feature incorporated into NNI 3.0.
As for now, NNI compressors are principally classified into three categories, namely pruner, quantizer, and distiller.
This new feature enables the compression of a single model by multiple compressors simultaneously.

For instance, users can apply varied pruning algorithms to different modules within the model,
along with training-aware quantization for model quantization.
Additionally, to maintain accuracy, relevant distillation techniques can be introduced.

.. Note::

    NNI strives to ensure maximum compatibility among different compressors in fusion compression.
    Nevertheless, it is impossible to avoid mutual interference in model modification between different compression algorithms in some individual scenarios.
    We encourage users to integrate algorithms after acquiring a comprehensive understanding of the fundamental principles of compression methods.
    If you encounter any problems or doubts that cannot be resolved while using fusion compression, you are welcome to raise an issue for discussion.

Main API
--------

To explain how fusion compression worked, we should know that each module in the model has a corresponding wrapper in the compressor.
The wrapper stores the necessary data required for compression.
After wrapping the original module with the wrapper, when need to execute ``module.forward``,
compressor will execute ``Wrapper.forward`` with simulated compression logic instead.

All compressors implement the class method ``from_compressor`` that can initialize a new compressor from the old ones.
The compressor initialized using this API will reuse the existing wrappers and record the preceding compression logic.
Multiple compressors can be initialized sequentially in the following format:
``fusion_compressor = Pruner.from_compressor(Quantizer.from_compressor(Distiller.from_compressor))``.

In general, the arguments of ``Compressor.from_compressor`` are mostly identical to the initialization arguments of the compressor.
The only difference is that the first argument of the initialization function is generally the model,
while the first parameter of ``from_compressor`` is typically one compressor object.
Additionally, if the fused compressor has no configured evaluator, one evaluator must be passed in ``from_compressor``.
However, if the evaluator has already in fused compressor, there is no need for duplicate passed in (it will be ignored if duplicated).

Example
-------

Pruning + Distillation
^^^^^^^^^^^^^^^^^^^^^^

The full example can be found `here <https://github.com/microsoft/nni/tree/master/examples/compression/fusion/pqd_fuse.py>`__.

The following code is a common pipeline with pruning first and then distillation.

.. code-block:: python

    ...
    pruner = Pruner(model, config_list, evaluator, ...)
    pruner.compress(max_steps, max_epochs)
    pruner.unwrap_model()

    masks = pruner.get_masks()
    model = ModelSpeedup(model, dummy_input, masks).speedup_model()
    ...
    distiller = Distiller(model, config_list, evaluator, teacher_model, teacher_predict, ...)
    distiller.compress(max_steps, max_epochs)

When attempting to implement a large sparsity, the reduction in accuracy post-pruning may become more pronounced,
necessitating greater exertion during the fine-tuning phase. The fusion of distillation and pruning can significantly mitigate this issue.  

The following code combines the pruner and distiller, resulting in a fusion compression.

.. code-block:: python

    ...
    pruner = Pruner(model, pruning_config_list, evaluator, ...)
    distiller = Distiller.from_compressor(pruner, distillation_config_list, teacher_model, teacher_predict, ...)
    distiller.compress(max_steps, max_epochs)

    masks = pruner.get_masks()
    model = ModelSpeedup(model, dummy_input, masks).speedup_model()

Also you could fuse any compressors you like by ``from_compressor``.

.. code-block:: python

    ...
    pruner_a = PrunerA(model, pruning_config_list_a, evaluator, ...)
    pruner_b = PrunerB.from_compressor(pruner_a, pruning_config_list_b, ...)
    pruner_c = PrunerC.from_compressor(pruner_b, pruning_config_list_c, ...)
    distiller_a = DistillerA.from_compressor(pruner_c, distillation_config_list_a, teacher_model, teacher_predict, ...)
    distiller_b = DistillerB.from_compressor(distiller_a, distillation_config_list_b, teacher_model, teacher_predict, ...)
    distiller_b.compress(max_steps, max_epochs)

    masks = pruner_c.get_masks()
    model = ModelSpeedup(model, dummy_input, masks).speedup_model()
