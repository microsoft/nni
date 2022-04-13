.. 5f266ace988c9ca9e44555fdc497e9ba

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_tutorials_pruning_quick_start_mnist.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_pruning_quick_start_mnist.py:


模型剪枝入门
============

模型剪枝是一种通过减小模型权重规模或中间状态规模来减小模型大小和计算量的技术。
修剪 DNN 模型有三种常见做法：

#. 训练一个模型 -> 对模型进行剪枝 -> 对剪枝后模型进行微调
#. 在模型训练过程中进行剪枝 -> 对剪枝后模型进行微调
#. 对模型进行剪枝 -> 从头训练剪枝后模型

NNI 主要通过在剪枝阶段进行工作来支持上述所有剪枝过程。
通过本教程可以快速了解如何在常见实践中使用 NNI 修剪模型。

.. GENERATED FROM PYTHON SOURCE LINES 17-22

准备工作
--------

在本教程中，我们使用一个简单的模型在 MNIST 数据集上进行了预训练。
如果你熟悉在 pytorch 中定义模型和训练模型，可以直接跳到 `模型剪枝`_。

.. GENERATED FROM PYTHON SOURCE LINES 22-35

.. code-block:: default


    import torch
    import torch.nn.functional as F
    from torch.optim import SGD

    from scripts.compression_mnist_model import TorchModel, trainer, evaluator, device

    # define the model
    model = TorchModel().to(device)

    # show the model structure, note that pruner will wrap the model layer.
    print(model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    TorchModel(
      (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=256, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )




.. GENERATED FROM PYTHON SOURCE LINES 36-47

.. code-block:: default


    # define the optimizer and criterion for pre-training

    optimizer = SGD(model.parameters(), 1e-2)
    criterion = F.nll_loss

    # pre-train and evaluate the model on MNIST dataset
    for epoch in range(3):
        trainer(model, optimizer, criterion)
        evaluator(model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Average test loss: 0.5266, Accuracy: 8345/10000 (83%)
    Average test loss: 0.2713, Accuracy: 9209/10000 (92%)
    Average test loss: 0.1919, Accuracy: 9356/10000 (94%)




.. GENERATED FROM PYTHON SOURCE LINES 48-58

模型剪枝
--------

使用 L1NormPruner 对模型进行剪枝并生成掩码。
通常情况下，pruner 需要原始模型和一个 ``config_list`` 作为输入参数。
具体关于如何写 ``config_list`` 请参考 :doc:`compression config specification <../compression/compression_config_list>`。

以下 `config_list` 表示 pruner 将修剪类型为 `Linear` 或 `Conv2d` 的所有层除了名为 `fc3` 的层，因为 `fc3` 被设置为 `exclude`。
每层的最终稀疏率是 50%。而名为 `fc3` 的层将不会被修剪。

.. GENERATED FROM PYTHON SOURCE LINES 58-67

.. code-block:: default


    config_list = [{
        'sparsity_per_layer': 0.5,
        'op_types': ['Linear', 'Conv2d']
    }, {
        'exclude': True,
        'op_names': ['fc3']
    }]








.. GENERATED FROM PYTHON SOURCE LINES 68-69

Pruners usually require `model` and `config_list` as input arguments.

.. GENERATED FROM PYTHON SOURCE LINES 69-76

.. code-block:: default


    from nni.compression.pytorch.pruning import L1NormPruner
    pruner = L1NormPruner(model, config_list)

    # show the wrapped model structure, `PrunerModuleWrapper` have wrapped the layers that configured in the config_list.
    print(model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    TorchModel(
      (conv1): PrunerModuleWrapper(
        (module): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
      )
      (conv2): PrunerModuleWrapper(
        (module): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      )
      (fc1): PrunerModuleWrapper(
        (module): Linear(in_features=256, out_features=120, bias=True)
      )
      (fc2): PrunerModuleWrapper(
        (module): Linear(in_features=120, out_features=84, bias=True)
      )
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )




.. GENERATED FROM PYTHON SOURCE LINES 77-84

.. code-block:: default


    # compress the model and generate the masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    conv1  sparsity :  0.5
    conv2  sparsity :  0.5
    fc1  sparsity :  0.5
    fc2  sparsity :  0.5




.. GENERATED FROM PYTHON SOURCE LINES 85-88

使用 NNI 的模型加速功能和 pruner 生成好的 masks 对原始模型进行加速，注意 `ModelSpeedup` 需要 unwrapped 的模型。
模型会在加速之后真正的在规模上变小，并且可能会达到相比于 masks 更大的稀疏率，这是因为 `ModelSpeedup` 会自动在模型中传播稀疏，
识别由于掩码带来的冗余权重。

.. GENERATED FROM PYTHON SOURCE LINES 88-97

.. code-block:: default


    # need to unwrap the model, if the model is wrapped before speedup
    pruner._unwrap_model()

    # speedup the model
    from nni.compression.pytorch.speedup import ModelSpeedup

    ModelSpeedup(model, torch.rand(3, 1, 28, 28).to(device), masks).speedup_model()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    aten::log_softmax is not Supported! Please report an issue at https://github.com/microsoft/nni. Thanks~
    Note: .aten::log_softmax.12 does not have corresponding mask inference object
    /home/ningshang/anaconda3/envs/nni-dev/lib/python3.8/site-packages/torch/_tensor.py:1013: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at  aten/src/ATen/core/TensorBody.h:417.)
      return self._grad




.. GENERATED FROM PYTHON SOURCE LINES 98-99

模型在加速之后变小了。

.. GENERATED FROM PYTHON SOURCE LINES 99-101

.. code-block:: default

    print(model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    TorchModel(
      (conv1): Conv2d(1, 3, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(3, 8, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=128, out_features=60, bias=True)
      (fc2): Linear(in_features=60, out_features=42, bias=True)
      (fc3): Linear(in_features=42, out_features=10, bias=True)
    )




.. GENERATED FROM PYTHON SOURCE LINES 102-106

微调压缩好的紧凑模型
--------------------

注意当前的模型已经经过了加速，如果你需要微调模型，请重新生成 optimizer。
这是因为在加速过程中进行了层替换，原来的 optimizer 已经不适用于现在的新模型了。

.. GENERATED FROM PYTHON SOURCE LINES 106-110

.. code-block:: default


    optimizer = SGD(model.parameters(), 1e-2)
    for epoch in range(3):
        trainer(model, optimizer, criterion)








.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  24.976 seconds)


.. _sphx_glr_download_tutorials_pruning_quick_start_mnist.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: pruning_quick_start_mnist.py <pruning_quick_start_mnist.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: pruning_quick_start_mnist.ipynb <pruning_quick_start_mnist.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
