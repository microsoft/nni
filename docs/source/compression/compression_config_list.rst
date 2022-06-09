Compression Config Specification
================================

Each sub-config in the config list is a dict, and the scope of each setting (key) is only internal to each sub-config.
If multiple sub-configs are configured for the same layer, the later ones will overwrite the previous ones.

Common Keys in Config
---------------------

op_types
^^^^^^^^

The type of the layers targeted by this sub-config.
If ``op_names`` is not set in this sub-config, all layers in the model that satisfy the type will be selected.
If ``op_names`` is set in this sub-config, the selected layers should satisfy both type and name.

op_names
^^^^^^^^

The name of the layers targeted by this sub-config.
If ``op_types`` is set in this sub-config, the selected layer should satisfy both type and name.

exclude
^^^^^^^

The ``exclude`` and ``sparsity`` keyword are mutually exclusive and cannot exist in the same sub-config.
If ``exclude`` is set in sub-config, the layers selected by this config will not be compressed.

Special Keys for Pruning
------------------------

op_partial_names
^^^^^^^^^^^^^^^^

This key will share with `Quantization Config` in the future.

This key is for the layers to be pruned with names that have the same sub-string. NNI will find all names in the model,
find names that contain one of ``op_partial_names``, and append them into the ``op_names``.

sparsity_per_layer
^^^^^^^^^^^^^^^^^^

The sparsity ratio of each selected layer.

e.g., the ``sparsity_per_layer`` is 0.8 means each selected layer will mask 80% values on the weight.
If ``layer_1`` (500 parameters) and ``layer_2`` (1000 parameters) are selected in this sub-config,
then ``layer_1`` will be masked 400 parameters and ``layer_2`` will be masked 800 parameters.

total_sparsity
^^^^^^^^^^^^^^

The sparsity ratio of all selected layers, means that sparsity ratio may no longer be even between layers.

e.g., the ``total_sparsity`` is 0.8 means 80% of parameters in this sub-config will be masked.
If ``layer_1`` (500 parameters) and ``layer_2`` (1000 parameters) are selected in this sub-config,
then ``layer_1`` and ``layer_2`` will be masked a total of 1200 parameters,
how these total parameters are distributed between the two layers is determined by the pruning algorithm.

sparsity
^^^^^^^^

``sparsity`` is an old config key from the pruning v1, it has the same meaning as ``sparsity_per_layer``.
You can also use ``sparsity`` right now, but it will be deprecated in the future.

max_sparsity_per_layer
^^^^^^^^^^^^^^^^^^^^^^

This key is usually used with ``total_sparsity``. It limits the maximum sparsity ratio of each layer.

In ``total_sparsity`` example, there are 1200 parameters that need to be masked and all parameters in ``layer_1`` may be totally masked.
To avoid this situation, ``max_sparsity_per_layer`` can be set as 0.9, this means up to 450 parameters can be masked in ``layer_1``,
and 900 parameters can be masked in ``layer_2``.

Special Keys for Quantization
-----------------------------

quant_types
^^^^^^^^^^^

Currently, nni support three kind of quantization types: 'weight', 'input', 'output'.
It can be set as ``str`` or ``List[str]``.
Note that 'weight' and 'input' are always quantize together, e.g., ``['input', 'weight']``.

quant_bits
^^^^^^^^^^

Bits length of quantization, key is the quantization type set in ``quant_types``, value is the length,
eg. {'weight': 8}, when the type is int, all quantization types share same bits length.

quant_start_step
^^^^^^^^^^^^^^^^

Specific key for ``QAT Quantizer``. Disable quantization until model are run by certain number of steps,
this allows the network to enter a more stable.
State where output quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0.

Examples
--------

Suppose we want to compress the following model::

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            ...
    
First, we need to determine where to compress, use the following config list to specify all ``Conv2d`` modules and module named ``fc1``::

    config_list = [{'op_types': ['Conv2d']}, {'op_names': ['fc1']}]

Sometimes we may need to compress all modules of a certain type, except for a few special ones.
Writing all the module names is laborious at this point, we can use ``exclude`` to quickly specify the compression target modules::

    config_list = [{
        'op_types': ['Conv2d', 'Linear']
    }, {
        'exclude': True,
        'op_names': ['fc2']
    }]

The above two config lists are equivalent to the model we want to compress, they both use ``conv1``, ``conv2``, and ``fc1`` as compression targets.

Let's take a simple pruning config list example, pruning all ``Conv2d`` modules with 50% sparsity, and pruning ``fc1`` with 80% sparsity::

    config_list = [{
        'op_types': ['Conv2d'],
        'total_sparsity': 0.5
    }, {
        'op_names': ['fc1'],
        'total_sparsity': 0.8
    }]

Then if you want to try model quantization, here is a simple config list example::

    config_list = [{
        'op_types': ['Conv2d'],
        'quant_types': ['input', 'weight'],
        'quant_bits': {'input': 8, 'weight': 8}
    }, {
        'op_names': ['fc1'],
        'quant_types': ['input', 'weight'],
        'quant_bits': {'input': 8, 'weight': 8}
    }]
