使用 NNI Tuners 自动进行模型压缩
========================================

使用 NNI 能轻松实现自动模型压缩

首先，使用 NNI 压缩模型
---------------------------------

可使用 NNI 轻松压缩模型。 以剪枝为例，可通过 L2FilterPruner 对预训练模型剪枝：

.. code-block:: python

   from nni.algorithms.compression.pytorch.pruning import L2FilterPruner
   config_list = [{ 'sparsity': 0.5, 'op_types': ['Conv2d'] }]
   pruner = L2FilterPruner(model, config_list)
   pruner.compress()

op_type 'Conv2d' 表示在 PyTorch 框架下定义在 :githublink:`default_layers.py <nni/compression/pytorch/default_layers.py>` 中的模块类型。

因此 ``{ 'sparsity': 0.5, 'op_types': ['Conv2d'] }`` 表示 **所有指定 op_types 的层都会被压缩到 0.5 的稀疏度**。 当调用 ``pruner.compress()`` 时，模型会通过掩码进行压缩。随后还可以微调模型，此时 **被剪除的权重不会被更新**。

然后，进行自动化
-------------------------

上一个示例手动选择 L2FilterPruner 并使用指定的稀疏度进行剪枝。 不同的稀疏度和不同的 Pruner 对不同的模型可能有不同的影响。 这个过程可以通过 NNI Tuner 完成。

首先，修改几行代码

.. code-block:: python

    import nni
    from nni.algorithms.compression.pytorch.pruning import *
   
    params = nni.get_parameters()
    sparsity = params['sparsity']
    pruner_name = params['pruner']
    model_name = params['model']

    model, pruner = get_model_pruner(model_name, pruner_name, sparsity)
    pruner.compress()

    train(model)  # 微调模型的代码
    acc = test(model)  # 测试微调后的模型
    nni.report_final_results(acc)

然后，在 YAML 中定义一个 ``config`` 文件来自动调整模型、剪枝算法和稀疏度。

.. code-block:: yaml

    searchSpace:
    sparsity:
      _type: choice
      _value: [0.25, 0.5, 0.75]
    pruner:
      _type: choice
      _value: ['slim', 'l2filter', 'fpgm', 'apoz']
    model:
      _type: choice
      _value: ['vgg16', 'vgg19']
    trainingService:
    platform: local
    trialCodeDirectory: .
    trialCommand: python3 basic_pruners_torch.py --nni
    trialConcurrency: 1
    trialGpuNumber: 0
    tuner:
      name: grid

完整实验代码在 :githublink:`这里 <examples/model_compress/pruning/config.yml>`

最后，开始搜索

.. code-block:: bash

   nnictl create -c config.yml
