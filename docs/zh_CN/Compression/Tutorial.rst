教程
========

.. contents::

在本教程中，我们将更详细地解释 NNI 中模型压缩的用法。 

设定压缩目标
----------------------

指定配置
^^^^^^^^^^^^^^^^^^^^^^^^^

用户可为压缩算法指定配置 (即, ``config_list`` )。 例如，压缩模型时，用户可能希望指定稀疏率，为不同类型的操作指定不同的稀疏比例，排除某些类型的操作，或仅压缩某类操作。 配置规范可用于表达此类需求。 可将其视为一个 Python 的 ``list`` 对象，其中每个元素都是一个 ``dict`` 对象。 

``list`` 中的 ``dict`` 会依次被应用，也就是说，如果一个操作出现在两个配置里，后面的 ``dict`` 会覆盖前面的配置。 

``dict`` 中有不同的键值。 以下是所有压缩算法都支持的：

* **op_types**：指定要压缩的操作类型。 'default' 表示使用算法的默认设置。
* **op_names**：指定需要压缩的操作的名称。 如果没有设置此字段，操作符不会通过名称筛选。
* **exclude**：默认为 False。 如果此字段为 True，表示要通过类型和名称，将一些操作从压缩中排除。

其他一些键值通常是针对某个特定算法的，可参考 `剪枝算法 <./Pruner.rst>`__ 和 `量化算法 <./Quantizer.rst>`__，查看每个算法的键值。

配置的简单示例如下：

.. code-block:: python

   [
       {
           'sparsity': 0.8,
           'op_types': ['default']
       },
       {
           'sparsity': 0.6,
           'op_names': ['op_name1', 'op_name2']
       },
       {
           'exclude': True,
           'op_names': ['op_name3']
       }
   ]

其表示压缩操作的默认稀疏度为 0.8，但 ``op_name1`` 和 ``op_name2`` 会使用 0.6，且不压缩 ``op_name3``。

量化算法特定键
^^^^^^^^^^^^^^^^^^^^^^^^^^

如果使用量化算法，则需要设置下面的 ``config_list``。 如果使用剪枝算法，则可以忽略这些键值。

* **quant_types** : 字符串列表。 

要应用量化的类型，当前支持 "权重"，"输入"，"输出"。 "权重"是指将量化操作
应用到 module 的权重参数上。 "输入" 是指对 module 的 forward 方法的输入应用量化操作。 "输出"是指将量化运法应用于模块 forward 方法的输出，在某些论文中，这种方法称为"激活"。


* **quant_bits**：int 或 dict {str : int}

量化的位宽，键是量化类型，值是量化位宽度，例如： 

.. code-block:: bash

   {
       quant_bits: {
           'weight': 8,
           'output': 4,
           },
   }

当值为 int 类型时，所有量化类型使用相同的位宽。 例如： 

.. code-block:: bash

   {
       quant_bits: 8, # 权重和输出的位宽都为 8 bits
   }

下面的示例展示了一个更完整的 ``config_list``，它使用 ``op_names``（或者 ``op_types``）指定目标层以及这些层的量化位数。

.. code-block:: bash

   config_list = [{
           'quant_types': ['weight'],        
           'quant_bits': 8, 
           'op_names': ['conv1']
       }, {
           'quant_types': ['weight'],
           'quant_bits': 4,
           'quant_start_step': 0,
           'op_names': ['conv2']
       }, {
           'quant_types': ['weight'],
           'quant_bits': 3,
           'op_names': ['fc1']
           },
          {
           'quant_types': ['weight'],
           'quant_bits': 2,
           'op_names': ['fc2']
           }
   ]

在这个示例中，'op_names' 是层的名字，四个层将被量化为不同的 quant_bits。


导出压缩结果
-------------------------

导出裁剪后的模型
^^^^^^^^^^^^^^^^^^^^^^^

使用下列 API 可轻松将裁剪后的模型导出，稀疏模型权重的 ``state_dict`` 会保存在 ``model.pth`` 文件中，可通过 ``torch.load('model.pth')`` 加载。 注意，导出的 ``model.pth`` 具有与原始模型相同的参数，只是掩码的权重为零。 ``mask_dict`` 存储剪枝算法产生的二进制值，可以进一步用来加速模型。

.. code-block:: python

   # 导出模型的权重和掩码。
   pruner.export_model(model_path='model.pth', mask_path='mask.pth')

   # 将掩码应用到模型
   from nni.compression.pytorch import apply_compression_results

   apply_compression_results(model, mask_file, device)


用 ``onnx`` 格式导出模型，（需要指定\ ``input_shape`` ）：

.. code-block:: python

   pruner.export_model(model_path='model.pth', mask_path='mask.pth', onnx_path='model.onnx', input_shape=[1, 1, 28, 28])


导出量化后的模型
^^^^^^^^^^^^^^^^^^^^^^^^^^

您可以使用 ``torch.save`` api 直接导出量化模型。量化后的模型可以通过 ``torch.load`` 加载，不需要做任何额外的修改。 下面的例子展示了使用 QAT quantizer 保存、加载量化模型并获取相关参数的过程。

.. code-block:: python
   
   # 保存使用 NNI QAT 算法生成的量化模型
   torch.save(model.state_dict(), "quantized_model.pth")

   # 模拟模型加载过程
   # 初始化新模型并在加载之前压缩它
   qmodel_load = Mnist()
   optimizer = torch.optim.SGD(qmodel_load.parameters(), lr=0.01, momentum=0.5)
   quantizer = QAT_Quantizer(qmodel_load, config_list, optimizer)
   quantizer.compress()
   
   # 加载量化的模型
   qmodel_load.load_state_dict(torch.load("quantized_model.pth"))

   # 获取加载后模型的 scale, zero_point 和 conv1 的权重
   conv1 = qmodel_load.conv1
   scale = conv1.module.scale
   zero_point = conv1.module.zero_point
   weight = conv1.module.weight


模型加速
------------------

掩码实际上并不能加速模型。 应该基于导出的掩码来对模型加速，因此，NNI 提供了 API 来加速模型。 在模型上调用 ``apply_compression_results`` 后，模型会变得更小，推理延迟也会减小。

.. code-block:: python

   from nni.compression.pytorch import apply_compression_results, ModelSpeedup

   dummy_input = torch.randn(config['input_shape']).to(device)
   m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
   m_speedup.speedup_model()


参考 `这里 <ModelSpeedup.rst>`__，了解详情。 模型加速的示例代码在 :githublink:`这里 <examples/model_compress/pruning/model_speedup.py>`。


控制微调过程
-------------------------------

控制微调的 API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

某些压缩算法会控制微调过程中的压缩进度（例如， `AGP <../Compression/Pruner.rst#agp-pruner>`__），一些算法需要在每个批处理步骤后执行一些逻辑。 因此，NNI 提供了两个 API：``pruner.update_epoch(epoch)`` 和 ``pruner.step()``。

``update_epoch`` 会在每个 Epoch 时调用，而 ``step`` 会在每次批处理后调用。 注意，大多数算法不需要调用这两个 API。 详细情况可参考具体算法文档。 对于不需要这两个 API 的算法，可以调用它们，但不会有实际作用。

强化微调过程
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

知识蒸馏有效地从大型教师模型中学习小型学生模型。 用户可以通过知识蒸馏来增强模型的微调过程，提高压缩模型的性能。 示例代码在 :githublink:`这里 <examples/model_compress/pruning/finetune_kd_torch.py>`。
