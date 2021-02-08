Quick Start
===========

.. contents::

Model compression usually consists of three stages: 1) pre-training a model, 2) compress the model, 3) fine-tuning the model. NNI mainly focus on the second stage that provides very simple APIs for compressing a model. Follow this guide for a quick look at how easy it is to use NNI to compress the model. 


Quick Start to Compress a Model
-------------------------------

To compress a model, both pruning and quantization can be applied. NNI provides a unified usage of them. Thus, here we use `level pruner <../Compression/Pruner.rst#level-pruner>`__ as an example to show the usage.

Step1. Write configuration
^^^^^^^^^^^^^^^^^^^

Write a configuration to specify the layers that you want to prune. The following configuration means pruning all the ``default``\ ops to sparsity 0.5 while keeping other layers unpruned.

.. code-block:: python

   config_list = [{
       'sparsity': 0.5,
       'op_types': ['default'],
   }]

The specification of configuration can be found `here <#specification-of-config-list>`__. Note that different pruners may have their own defined fields in configuration, for exmaple ``start_epoch`` in AGP pruner. Please refer to each pruner's `usage <./Pruner.rst>`__ for details, and adjust the configuration accordingly.

Step2. Choose a pruner and compress the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First instantiate the chosen pruner with your model and configuration as arguments, then invoke ``compress()`` to compress your model.

.. code-block:: python

   from nni.algorithms.compression.pytorch.pruning import LevelPruner

   optimizer_finetune = torch.optim.SGD(model.parameters(), lr=0.01)
   pruner = LevelPruner(model, config_list, optimizer_finetune)
   model = pruner.compress()

Then, you can train your model using traditional training approach (e.g., SGD), pruning is applied transparently during the training. Some pruners prune once at the beginning, the following training can be seen as fine-tune. Some pruners prune your model iteratively, the masks are adjusted epoch by epoch during training.

Step3. Export compression result
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After training, you can export model weights to a file, and the generated masks to a file as well. Exporting onnx model is also supported.

.. code-block:: python

   pruner.export_model(model_path='pruned_vgg19_cifar10.pth', mask_path='mask_vgg19_cifar10.pth')

Plese refer to :githublink:`mnist example <examples/model_compress/pruning/naive_prune_torch.py>` for example code.

Congratulations! You've compressed your first model via NNI. To go a bit more in depth and learn more about NNI compressino, check out the Tutorials.
