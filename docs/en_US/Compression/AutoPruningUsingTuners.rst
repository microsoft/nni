Automatic Model Pruning using NNI Tuners
========================================

It's convenient to implement auto model pruning with NNI compression and NNI tuners

First, model compression with NNI
---------------------------------

You can easily compress a model with NNI compression. Take pruning for example, you can prune a pretrained model with L2FilterPruner like this

.. code-block:: python

   from nni.algorithms.compression.pytorch.pruning import L2FilterPruner
   config_list = [{ 'sparsity': 0.5, 'op_types': ['Conv2d'] }]
   pruner = L2FilterPruner(model, config_list)
   pruner.compress()

The 'Conv2d' op_type stands for the module types defined in :githublink:`default_layers.py <nni/compression/pytorch/default_layers.py>` for pytorch.

Therefore ``{ 'sparsity': 0.5, 'op_types': ['Conv2d'] }``\ means that **all layers with specified op_types will be compressed with the same 0.5 sparsity**. When ``pruner.compress()`` called, the model is compressed with masks and after that you can normally fine tune this model and **pruned weights won't be updated** which have been masked.

Then, make this automatic
-------------------------

The previous example manually chose L2FilterPruner and pruned with a specified sparsity. Different sparsity and different pruners may have different effects on different models. This process can be done with NNI tuners.

Firstly, modify our codes for few lines

.. code-block:: python

    import nni
    from nni.algorithms.compression.pytorch.pruning import *
   
    params = nni.get_parameters()
    sparsity = params['sparsity']
    pruner_name = params['pruner']
    model_name = params['model']

    model, pruner = get_model_pruner(model_name, pruner_name, sparsity)
    pruner.compress()

    train(model)  # your code for fine-tuning the model
    acc = test(model)  # test the fine-tuned model
    nni.report_final_results(acc)

Then, define a ``config`` file in YAML to automatically tuning model, pruning algorithm and sparsity.

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

The full example can be found :githublink:`here <examples/model_compress/pruning/config.yml>`

Finally, start the searching via

.. code-block:: bash

   nnictl create -c config.yml
