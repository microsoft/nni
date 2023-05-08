Compression Config Specification
================================

Common Keys in Config
---------------------

op_names
^^^^^^^^

A list of fully-qualified name of modules (e.g., ``['backbone.layers.0.ffn', ...]``) that will be compressed.
If the name referenced module is not existed in the model, it will be ignored.

op_names_re
^^^^^^^^^^^

A list of regular expressions for matching module names by python standard library ``re``.
The matched modules will be selected to be compressed.

op_types
^^^^^^^^

A list of type names of classes that inherit from ``torch.nn.Module``.
Only module types in this list can be selected to be compressed.
If this key is not set, all module types can be selected.
If neither ``op_names`` or ``op_names_re`` are set, all modules satisfied the ``op_types`` are selected.

exclude_op_names
^^^^^^^^^^^^^^^^

A list of fully-qualified name of modules that are excluded.

exclude_op_names_re
^^^^^^^^^^^^^^^^^^^

A list of regular expressions for matching module names.
The matched modules will be removed from the modules that need to be compressed.

exclude_op_types
^^^^^^^^^^^^^^^^

A list of type names of classes that inherit from ``torch.nn.Module``.
The module types in this list are excluded from compression.

.. _target_names:

target_names
^^^^^^^^^^^^

A list of legal compression target name, i.e., usually ``_input_``, ``weight``, ``bias``, ``_output_`` are support to be compressed.

Two kinds of target are supported by design, module inputs/outputs(should be a tensor), module parameters:

- Inputs/Outputs: If the module inputs or outputs is a singal tensor, directly set ``_input_`` for input and ``_output_`` for output.
  ``_input_{position_index}`` or ``_input_{arg_name}`` can be used to specify the input target,
  i.e., for a forward function ``def forward(self, x: Tensor, y: Tensor, z: Any): ...``, ``_input_0`` or ``_input_x`` can be used to specify ``x`` to be compressed,
  note that ``self`` will be ignored when counting the position index.
  Similarly, ``_output_{position_index}`` can be used to specify the output target if the output is a ``list/tuple``,
  ``_output_{dict_key}`` can be used to specify the output target if the output is a ``dict``.
- Parameters/Buffers: Directly using the attribute name to specify the target, i.e., ``weight``, ``bias``.

.. _target_settings:

target_settings
^^^^^^^^^^^^^^^

A ``dict`` of target settings, the format is ``{target_name: setting}``. Target setting usually configure how to compress the target.

All other keys(except these eight common keys) in a config will seems as a shortcut of target setting key, and will apply to all targets selected in this config.
For example, consider a model has two ``Linear`` module (linear module names are ``'fc1'`` and ``'fc2'``), the following configs have same effect for pruning.

.. code-block:: python

    shorthand_config = {
        'op_types': ['Linear'],
        'sparse_ratio': 0.8
    }

    standard_config = {
        'op_names': ['fc1', 'fc2'],
        'target_names': ['weight', 'bias'],
        'target_settings': {
            'weight': {
                'sparse_ratio': 0.8,
                'max_sparse_ratio': None,
                'min_sparse_ratio': None,
                'sparse_threshold': None,
                'global_group_id': None,
                'dependency_group_id': None,
                'granularity': 'default',
                'internal_metric_block': None,
                'apply_method': 'mul',
            },
            'bias': {
                'align': {
                    'target_name': 'weight',
                    'dims': [0],
                },
                'apply_method': 'mul',
            }
        }
    }


.. Note:: Each compression target can only be configure once, re-configuration will not take effect.

Pruning Specific Configuration Keys
-----------------------------------

sparse_ratio
^^^^^^^^^^^^

A float number between 0. ~ 1., the sparse ratio of the pruning target or the total sparse ratio of a group of pruning targets.
For example, if the sparse ratio is 0.8, and the pruning target is a Linear module weight, 80% weight value will be masked after pruning.

max_sparse_ratio
^^^^^^^^^^^^^^^^

This key is usually used in combination with ``sparse_threshold`` and ``global_group_id``, limit the maximum sparse ratio of each target.

A float number between 0. ~ 1., for each single pruning target, the sparse ratio after pruning will not be larger than this number,
that means at most masked ``max_sparse_ratio`` pruning target value.

min_sparse_ratio
^^^^^^^^^^^^^^^^

This key is usually used in combination with ``sparse_threshold`` and ``global_group_id``, limit the minimum sparse ratio of each target.

A float number between 0. ~ 1., for each single pruning target, the sparse ratio after pruning will not be lower than this number,
that means at least masked ``min_sparse_ratio`` pruning target value.

sparse_threshold
^^^^^^^^^^^^^^^^

A float number, different from the ``sparse_ratio`` which configures a specific sparsity, ``sparse_threshold`` usually used in some adaptive sparse cases.
``sparse_threshold`` is directly compared to pruning metrics (different in different algorithms) and the positions smaller than the threshold are masked.

The value range is different for different pruning algorithms, please reference the pruner document to see how to configure it.
In general, the higher the threshold, the higher the final sparsity. 

.. _global_group_id:

global_group_id
^^^^^^^^^^^^^^^

``global_group_id`` should jointly used with ``sparse_ratio``.
All pruning targets that have same ``global_group_id`` will be treat as a whole, and the ``sparse_ratio`` will be distributed across pruning targets.
That means each pruning target might have different sparse ratio after pruning, but the group sparse ratio will be the configured ``sparse_ratio``.

Note that the ``sparse_ratio`` in the same global group should be the same.

For example, a model has three ``Linear`` modules (``'fc1'``, ``'fc2'``, ``'fc3'``),
and the expected total sparse ratio of these three modules is 0.5, then the config can be:

.. code-block:: python

    config_list = [{
        'op_names': ['fc1', 'fc2'],
        'sparse_ratio': 0.5,
        'global_group_id': 'linear_group_1'
    }, {
        'op_names': ['fc3'],
        'sparse_ratio': 0.5,
        'global_group_id': 'linear_group_1'
    }]


.. _dependency_group_id:

dependency_group_id
^^^^^^^^^^^^^^^^^^^

All pruning targets that have same ``dependency_group_id`` will be treat as a whole, and the positions the targets' pruned will be the same.
For example, layer A and layer B have same ``dependency_group_id``, and they want to be pruned output channels, then A and B will be pruned the same channel indexes.

Note that the ``sparse_ratio`` in the same dependency group should be the same, and the prunable positions (after reduction by ``granularity``) should be same,
for example, pruning targets should have same output channel number when pruning output channel.

This key usually be used on modules with add or mul operation, i.e., skip connection.

If you don't know your model structure well, you could use :ref:`auto_set_denpendency_group_ids` to auto detect the dependency operations and auto set their ``dependency_group_id``.

granularity
^^^^^^^^^^^

Control the granularity of the generated masked.

``default``, ``in_channel``, ``out_channel``, ``per_channel`` and list of integer are supported:

- default: The pruner will auto determine using which kind of granularity, usually consistent with the paper.
- in_channel: The pruner will do pruning on the weight parameters 1 dimension.
- out_channel: The pruner will do pruning on the weight parameters 0 dimension.
- per_channel: The pruner will do pruning on the input/output -1 dimension.
- list of integer: Block sparse will be applied. For example, ``[4, 4]`` will apply 4x4 block sparse on the last two dimensions of the weight parameters.

Note that ``in_channel`` or ``out_channel`` is not supported for input/output targets, please using ``per_channel`` instead.
``torch.nn.Embedding`` is special, it's output dimension on weight is 1, so if want to pruning Embedding output channel, please set ``in_channel`` for its granularity for workaround.

The following is an example for output channel pruning:

.. code-block:: python

    config = {
        'op_types': ['Conv2d'],
        'sparse_ratio': 0.5,
        'granularity': 'out_channel' # same as [1, -1, -1, -1]
    }

.. _align:

align
^^^^^

``align`` refers to the process where the target mask will not be generated by the pruning algorithm but is created in accordance with another pruning target mask.

A typical scenario occurs in most PyTorch native modules with ``weight`` and ``bias`` attributes.
In this case, the generation of the ``bias`` mask is aligned with the ``weight`` mask generation,
meaning that a ``bias`` value is masked or not depending on whether the related ``weight`` values are all masked or not.
For example, in all pruners, a ``bias`` in a ``Linear`` layer masks the ``i`` position when the ``i`` row values in ``weight`` are all masked.

This can also prove useful for generating activation masks (output of activation modules).
For instance, consider the common pattern in transformers: ``conv-bn-relu``.
Here, the ``bn`` weight or output can be masked in alignment with the convolution weight for a more effective pruning simulation.

.. code-block:: python

    config = {
        'op_types': ['BatchNorm2d'],
        'target_names': ['weight'],
        'target_settings': {
            'weight': {
                'align': {
                    'module_name': 'conv',
                    'target_name': 'weight',
                    'dims': [0],
                }
            }
        }
    }

The mentioned configuration implies that the batch normalization layer should align with the ``weight`` of the ``conv`` layer along dimension ``0``.

module_name
"""""""""""

By default, the current configured module. The name of the module that align with.

target_name
"""""""""""

Align with which tagret mask of the specified module.

dims
""""

Align the mask on which dim of the specified target.

apply_method
^^^^^^^^^^^^

By default, ``mul``. ``mul`` and ``add`` is supported to apply mask on pruning target.

``mul`` means the pruning target will be masked by multiply a mask metrix contains 0 and 1, 0 represents masked position, 1 represents unmasked position.

``add`` means the pruning target will be masked by add a mask metrix contains -1000 and 0, -1000 represents masked position, 0 represents unmasked position.
Note that -1000 can be configured in the future. ``add`` usually be used to mask activation module such as Softmax.

Quantization Specific Configuration Keys
----------------------------------------

quant_dtype
^^^^^^^^^^^

By default, ``int8``. Support ``int`` and ``uint`` plus quant bits.

quant_scheme
^^^^^^^^^^^^

``affine`` or ``symmetric``. If this key is not set, the quantization scheme will be choosen by quantizer,
most quantizer will apply ``symmetric`` quantization.

fuse_names
^^^^^^^^^^

``List[(str,)]``. Optional parameter, each tuple defines the module and modules that need to be fused in the first module. 
Each element in the tuple is the module name in the model. 
Note that the first module name in each tuple should be in the ``op_name`` or ``op_name_re``.

granularity
^^^^^^^^^^^

Used to control the granularity of the target quantization, by default the whole tensor will use the same scale and zero point.

``per_channel`` and list of integer are supported:

- ``per_channel``: Each (ouput) channel will have their independent scales and zero points.
- list of integer: The integer list is the block size. Each block will have their independent scales and zero points.

Each sub-config in the config list is a dict, and the scope of each setting (key) is only internal to each sub-config.
If multiple sub-configs are configured for the same layer, the later ones will overwrite the previous ones.

Distillation Specific Configuration Keys
----------------------------------------

lambda
^^^^^^

A float number. The scale factor of the distillation loss.
The final distil loss for the specific target is ``lambda * distil_loss_func(student_target, teacher_target)``.

link
^^^^

A teacher module name or a list of teacher module names. The student module link to.

apply_method
^^^^^^^^^^^^

``mse`` or ``kl``.

``mse`` means the MSE loss, usually used to distill hidden states.
Please reference `mse_loss <https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html>`__.

``kl`` means the KL loss, usually used to distill logits.
The implementation is ``kl_div((stu_hs / 2).log_softmax(dim=-1), (tea_hs / 2).softmax(dim=-1), reduction='batchmean') * (2 ** 2)``,
please reference `kl_div <https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html>`__.
