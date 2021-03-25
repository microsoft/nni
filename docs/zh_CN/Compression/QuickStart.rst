快速入门
===========

..  toctree::
    :hidden:

    教程 <Tutorial>


模型压缩通常包括三个阶段：1）预训练模型，2）压缩模型，3）微调模型。 NNI 主要关注于第二阶段，并为模型压缩提供非常简单的 API。 遵循本指南，快速了解如何使用 NNI 压缩模型。 

模型剪枝
-------------

这里通过 `level pruner <../Compression/Pruner.rst#level-pruner>`__ 举例说明 NNI 中模型剪枝的用法。

Step1. 编写配置
^^^^^^^^^^^^^^^^^^^^^^^^^^

编写配置来指定要剪枝的层。 以下配置表示剪枝所有的 ``default`` 操作，稀疏度设为 0.5，其它层保持不变。

.. code-block:: python

   config_list = [{
       'sparsity': 0.5,
       'op_types': ['default'],
   }]

配置说明在 `这里 <./Tutorial.rst#specify-the-configuration>`__。 注意，不同的 Pruner 可能有自定义的配置字段，例如，AGP Pruner 有 ``start_epoch``。 详情参考每个 Pruner 的 `用法 <./Pruner.rst>`__，来调整相应的配置。

Step2. 选择 Pruner 来压缩模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

首先，使用模型来初始化 Pruner，并将配置作为参数传入，然后调用 ``compress()`` 来压缩模型。 请注意，有些算法可能会检查压缩的梯度，因此我们还定义了一个优化器并传递给 Pruner。

.. code-block:: python

   from nni.algorithms.compression.pytorch.pruning import LevelPruner

   optimizer_finetune = torch.optim.SGD(model.parameters(), lr=0.01)
   pruner = LevelPruner(model, config_list, optimizer_finetune)
   model = pruner.compress()

然后，使用正常的训练方法来训练模型 （如，SGD），剪枝在训练过程中是透明的。 有些 Pruner（如 L1FilterPruner、FPGMPruner）在开始时修剪一次，下面的训练可以看作是微调。 有些 Pruner（例如AGPPruner）会迭代的对模型剪枝，在训练过程中逐步修改掩码。

注意，``pruner.compress`` 只会在模型权重上直接增加掩码，不包括调优的逻辑。 如果要想调优压缩后的模型，需要在 ``pruner.compress`` 后增加调优的逻辑。

例如：

.. code-block:: python

   for epoch in range(1, args.epochs + 1):
        pruner.update_epoch(epoch)
        train(args, model, device, train_loader, optimizer_finetune, epoch)
        test(model, device, test_loader)

更多关于微调的 API 在 `这里 <./Tutorial.rst#apis-to-control-the-fine-tuning>`__。 


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
       'quant_types': ['weight'],
       'quant_bits': {
           'weight': 8,
       }, # 这里可以仅使用 `int`，因为所有 `quan_types` 使用了一样的位长，参考下方 `ReLu6` 配置。
       'op_types':['Conv2d', 'Linear']
   }, {
       'quant_types': ['output'],
       'quant_bits': 8,
       'quant_start_step': 7000,
       'op_types':['ReLU6']
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

您可以使用 ``torch.save`` api 直接导出量化模型。量化后的模型可以通过 ``torch.load`` 加载，不需要做任何额外的修改。

.. code-block:: python

   # 保存使用 NNI QAT 算法生成的量化模型
   torch.save(model.state_dict(), "quantized_model.pth")

参考 :githublink:`mnist example <examples/model_compress/quantization/QAT_torch_quantizer.py>` 获取示例代码。

恭喜！ 您已经通过 NNI 压缩了您的第一个模型。 更深入地了解 NNI 中的模型压缩，请查看 `Tutorial <./Tutorial.rst>`__。