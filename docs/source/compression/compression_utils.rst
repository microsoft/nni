Analysis Utils for Model Compression
====================================

We provide several easy-to-use tools for users to analyze their model during model compression.

.. _topology-analysis:

Topology Analysis
-----------------

We also provide several tools for the topology analysis during the model compression. These tools are to help users compress their model better. Because of the complex topology of the network, when compressing the model, users often need to spend a lot of effort to check whether the compression configuration is reasonable. So we provide these tools for topology analysis to reduce the burden on users.

ChannelDependency
^^^^^^^^^^^^^^^^^

Complicated models may have residual connection/concat operations in their models. When the user prunes these models, they need to be careful about the channel-count dependencies between the convolution layers in the model. Taking the following residual block in the resnet18 as an example. The output features of the ``layer2.0.conv2`` and ``layer2.0.downsample.0`` are added together, so the number of the output channels of ``layer2.0.conv2`` and ``layer2.0.downsample.0`` should be the same, or there may be a tensor shape conflict.


.. image:: ../../img/channel_dependency_example.jpg
   :target: ../../img/channel_dependency_example.jpg
   :alt: 
 

If the layers have channel dependency are assigned with different sparsities (here we only discuss the structured pruning by L1FilterPruner/L2FilterPruner), then there will be a shape conflict during these layers. Even the pruned model with mask works fine, the pruned model cannot be speedup to the final model directly that runs on the devices, because there will be a shape conflict when the model tries to add/concat the outputs of these layers. This tool is to find the layers that have channel count dependencies to help users better prune their model.

Usage
"""""

.. code-block:: python

   from nni.compression.pytorch.utils.shape_dependency import ChannelDependency
   data = torch.ones(1, 3, 224, 224).cuda()
   channel_depen = ChannelDependency(net, data)
   channel_depen.export('dependency.csv')

Output Example
""""""""""""""

The following lines are the output example of torchvision.models.resnet18 exported by ChannelDependency. The layers at the same line have output channel dependencies with each other. For example, layer1.1.conv2, conv1, and layer1.0.conv2 have output channel dependencies with each other, which means the output channel(filters) numbers of these three layers should be same with each other, otherwise, the model may have shape conflict. 

.. code-block:: bash

   Dependency Set,Convolutional Layers
   Set 1,layer1.1.conv2,layer1.0.conv2,conv1
   Set 2,layer1.0.conv1
   Set 3,layer1.1.conv1
   Set 4,layer2.0.conv1
   Set 5,layer2.1.conv2,layer2.0.conv2,layer2.0.downsample.0
   Set 6,layer2.1.conv1
   Set 7,layer3.0.conv1
   Set 8,layer3.0.downsample.0,layer3.1.conv2,layer3.0.conv2
   Set 9,layer3.1.conv1
   Set 10,layer4.0.conv1
   Set 11,layer4.0.downsample.0,layer4.1.conv2,layer4.0.conv2
   Set 12,layer4.1.conv1

MaskConflict
^^^^^^^^^^^^

When the masks of different layers in a model have conflict (for example, assigning different sparsities for the layers that have channel dependency), we can fix the mask conflict by MaskConflict. Specifically, the MaskConflict loads the masks exported by the pruners(L1FilterPruner, etc), and check if there is mask conflict, if so, MaskConflict sets the conflicting masks to the same value.

.. code-block:: python

   from nni.compression.pytorch.utils.mask_conflict import fix_mask_conflict
   fixed_mask = fix_mask_conflict('./resnet18_mask', net, data)

not_safe_to_prune
^^^^^^^^^^^^^^^^^

If we try to prune a layer whose output tensor is taken as the input by a shape-constraint OP(for example, view, reshape), then such pruning maybe not be safe. For example, we have a convolutional layer followed by a view function.

.. code-block:: python

   x = self.conv(x) # output shape is (batch, 1024, 3, 3)
   x = x.view(-1, 1024)

If the output shape of the pruned conv layer is not divisible by 1024(for example(batch, 500, 3, 3)), we may meet a shape error. We cannot replace such a function that directly operates on the Tensor. Therefore, we need to be careful when pruning such layers. The function not_safe_to_prune finds all the layers followed by a shape-constraint function. Here is an example for usage. If you meet a shape error when running the forward inference on the speeduped model, you can exclude the layers returned by not_safe_to_prune and try again. 

.. code-block:: python

   not_safe = not_safe_to_prune(model, dummy_input)

.. _flops-counter:

Model FLOPs/Parameters Counter
------------------------------

We provide a model counter for calculating the model FLOPs and parameters. This counter supports calculating FLOPs/parameters of a normal model without masks, it can also calculates FLOPs/parameters of a model with mask wrappers, which helps users easily check model complexity during model compression on NNI. Note that, for sturctured pruning, we only identify the remained filters according to its mask, which not taking the pruned input channels into consideration, so the calculated FLOPs will be larger than real number (i.e., the number calculated after Model Speedup). 

We support two modes to collect information of modules. The first mode is ``default``\ , which only collect the information of convolution and linear. The second mode is ``full``\ , which also collect the information of other operations. Users can easily use our collected ``results`` for futher analysis.

Usage
^^^^^

.. code-block:: python

   from nni.compression.pytorch.utils import count_flops_params

   # Given input size (1, 1, 28, 28)
   flops, params, results = count_flops_params(model, (1, 1, 28, 28)) 

   # Given input tensor with size (1, 1, 28, 28) and switch to full mode
   x = torch.randn(1, 1, 28, 28)

   flops, params, results = count_flops_params(model, (x,), mode='full') # tuple of tensor as input

   # Format output size to M (i.e., 10^6)
   print(f'FLOPs: {flops/1e6:.3f}M,  Params: {params/1e6:.3f}M')
   print(results)
   {
   'conv': {'flops': [60], 'params': [20], 'weight_size': [(5, 3, 1, 1)], 'input_size': [(1, 3, 2, 2)], 'output_size': [(1, 5, 2, 2)], 'module_type': ['Conv2d']}, 
   'conv2': {'flops': [100], 'params': [30], 'weight_size': [(5, 5, 1, 1)], 'input_size': [(1, 5, 2, 2)], 'output_size': [(1, 5, 2, 2)], 'module_type': ['Conv2d']}
   }
