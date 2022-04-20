.. 98b0285bbfe1a01c90b9ba6a9b0d6caa

快速入门
===========

.. code-block::

   ..  toctree::
      :hidden:

      Notebook Example <compression_pipeline_example>


模型压缩通常包括三个阶段：1）预训练模型，2）压缩模型，3）微调模型。 NNI 主要关注于第二阶段，并为模型压缩提供易于使用的 API。遵循本指南，您将快速了解如何使用 NNI 来压缩模型。更深入地了解 NNI 中的模型压缩模块，请查看 `Tutorial <./Tutorial.rst>`__。

.. 提供了一个在 Jupyter notebook 中进行完整的模型压缩流程的 `示例 <./compression_pipeline_example.rst>`__，参考 :githublink:`代码 <examples/notebooks/compression_pipeline_example.ipynb>`。

模型剪枝
-------------

这里通过 `level pruner <../Compression/Pruner.rst#level-pruner>`__ 举例说明 NNI 中模型剪枝的用法。

Step1. 编写配置
^^^^^^^^^^^^^^^^^^^^^^^^^^

编写配置来指定要剪枝的层。以下配置表示剪枝所有的 ``default`` 层，稀疏度设为 0.5，其它层保持不变。

.. code-block:: python

   config_list = [{
       'sparsity': 0.5,
       'op_types': ['default'],
   }]

配置说明在 `这里 <./Tutorial.rst#specify-the-configuration>`__。注意，不同的 Pruner 可能有自定义的配置字段。详情参考每个 Pruner 的 `具体用法 <./Pruner.rst>`__，来调整相应的配置。

Step2. 选择 Pruner 来压缩模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

首先，使用模型来初始化 Pruner，并将配置作为参数传入，然后调用 ``compress()`` 来压缩模型。请注意，有些算法可能会检查训练过程中的梯度，因此我们可能会定义一组 trainer, optimizer, criterion 并传递给 Pruner。

.. code-block:: python

   from nni.algorithms.compression.pytorch.pruning import LevelPruner

   pruner = LevelPruner(model, config_list)
   model = pruner.compress()

然后，使用正常的训练方法来训练模型 （如，SGD），剪枝在训练过程中是透明的。有些 Pruner（如 L1FilterPruner、FPGMPruner）在开始时修剪一次，下面的训练可以看作是微调。有些 Pruner（例如AGPPruner）会迭代的对模型剪枝，在训练过程中逐步修改掩码。

如果使用 Pruner 进行迭代剪枝，或者剪枝过程中需要训练或者推理，则需要将 finetune 逻辑传到 Pruner 中。

例如：

.. code-block:: python

   from nni.algorithms.compression.pytorch.pruning import AGPPruner

   pruner = AGPPruner(model, config_list, optimizer, trainer, criterion, num_iterations=10, epochs_per_iteration=1, pruning_algorithm='level')
   model = pruner.compress()

Step3. 导出压缩结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

训练之后，可将模型权重导出到文件，同时将生成的掩码也导出到文件， 也支持导出 ONNX 模型。

.. code-block:: python

   pruner.export_model(model_path='pruned_vgg19_cifar10.pth', mask_path='mask_vgg19_cifar10.pth')

参考 :githublink:`mnist 示例 <examples/model_compress/pruning/naive_prune_torch.py>` 获取代码。

更多剪枝算法的示例在 :githublink:`basic_pruners_torch <examples/model_compress/pruning/basic_pruners_torch.py>` 和 :githublink:`auto_pruners_torch <examples/model_compress/pruning/auto_pruners_torch.py>`。


模型量化
------------------

这里通过 `QAT  Quantizer <../Compression/Quantizer.rst#qat-quantizer>`__ 举例说明在 NNI 中量化的用法。

Step1. 编写配置
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   config_list = [{
       'quant_types': ['weight', 'input'],
       'quant_bits': {
           'weight': 8,
           'input': 8,
       }, # 这里可以仅使用 `int`，因为所有 `quan_types` 使用了一样的位长，参考下方 `ReLu6` 配置。
       'op_types':['Conv2d', 'Linear'],
       'quant_dtype': 'int',
       'quant_scheme': 'per_channel_symmetric'
   }, {
       'quant_types': ['output'],
       'quant_bits': 8,
       'quant_start_step': 7000,
       'op_types':['ReLU6'],
       'quant_dtype': 'uint',
       'quant_scheme': 'per_tensor_affine'
   }]

配置说明在 `这里 <./Tutorial.rst#quantization-specific-keys>`__。

Step2. 选择 Quantizer 来压缩模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

   quantizer = QAT_Quantizer(model, config_list)
   quantizer.compress()


Step3. 导出压缩结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在训练和校准之后，你可以将模型权重导出到一个文件，并将生成的校准参数也导出到一个文件。 也支持导出 ONNX 模型。

.. code-block:: python

   calibration_config = quantizer.export_model(model_path, calibration_path, onnx_path, input_shape, device)

参考 :githublink:`mnist example <examples/model_compress/quantization/QAT_torch_quantizer.py>` 获取示例代码。

恭喜！ 您已经通过 NNI 压缩了您的第一个模型。 更深入地了解 NNI 中的模型压缩，请查看 `Tutorial <./Tutorial.rst>`__。