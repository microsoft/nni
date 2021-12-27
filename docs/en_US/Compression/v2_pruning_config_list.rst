Pruning Config Specification
============================

The Keys in Config List
-----------------------

Each sub-config in the config list is a dict, and the scope of each setting (key) is only internal to each sub-config.
If multiple sub-configs are configured for the same layer, the later ones will overwrite the previous ones.

op_types
^^^^^^^^

The type of the layers targeted by this sub-config.
If ``op_names`` is not set in this sub-config, all layers in the model that satisfy the type will be selected.
If ``op_names`` is set in this sub-config, the selected layers should satisfy both type and name.

op_names
^^^^^^^^

The name of the layers targeted by this sub-config.
If ``op_types`` is set in this sub-config, the selected layer should satisfy both type and name.

op_partial_names
^^^^^^^^^^^^^^^^

This key is for the layers to be pruned with names that have the same sub-string. NNI will find all names in the model,
find names that contain one of ``op_partial_names``, and append them into the ``op_names``.

sparsity
^^^^^^^^

The sparsity ratio of each selected layer.

e.g., the ``sparsity`` is 0.8 means each selected layer will mask 80% values on the weight.
If ``layer_1`` (500 parameters) and ``layer_2`` (1000 parameters) are selected in this sub-config,
then ``layer_1`` will be masked 400 parameters and ``layer_2`` will be masked 800 parameters.

sparsity_per_layer
^^^^^^^^^^^^^^^^^^

Another name for ``sparsity``.

total_sparsity
^^^^^^^^^^^^^^

The sparsity ratio of all selected layers, means that sparsity ratio may no longer be even between layers.

e.g., the ``total_sparsity`` is 0.8 means 80% of parameters in this sub-config will be masked.
If ``layer_1`` (500 parameters) and ``layer_2`` (1000 parameters) are selected in this sub-config,
then ``layer_1`` and ``layer_2`` will be masked a total of 1200 parameters,
how these total parameters are distributed between the two layers is determined by the pruning algorithm.

max_sparsity_per_layer
^^^^^^^^^^^^^^^^^^^^^^

This key is usually used with ``total_sparsity``. It limits the maximum sparsity ratio of each layer.

In ``total_sparsity`` example, there are 1200 parameters that need to be masked and all parameters in ``layer_1`` may be totally masked.
To avoid this situation, ``max_sparsity_per_layer`` can be set as 0.9, this means up to 450 parameters can be masked in ``layer_1``,
and 900 parameters can be masked in ``layer_2``.

exclude
^^^^^^^

This key cannot exist in a sub-config at the same time with keys containing ``sparsity``.
If ``exclude`` is set in sub-config, the layers selected by this config will not be pruned.
