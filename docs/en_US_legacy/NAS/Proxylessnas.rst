ProxylessNAS on NNI
===================

Introduction
------------

The paper `ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware <https://arxiv.org/pdf/1812.00332.pdf>`__ removes proxy, it directly learns the architectures for large-scale target tasks and target hardware platforms. They address high memory consumption issue of differentiable NAS and reduce the computational cost to the same level of regular training while still allowing a large candidate set. Please refer to the paper for the details.

Usage
-----

To use ProxylessNAS training/searching approach, users need to specify search space in their model using `NNI NAS interface <./MutationPrimitives.rst>`__\ , e.g., ``LayerChoice``\ , ``InputChoice``. After defining and instantiating the model, the following work can be leaved to ProxylessNasTrainer by instantiating the trainer and passing the model to it.

.. code-block:: python

   trainer = ProxylessTrainer(model,
                              loss=LabelSmoothingLoss(),
                              dataset=None,
                              optimizer=optimizer,
                              metrics=lambda output, target: accuracy(output, target, topk=(1, 5,)),
                              num_epochs=120,
                              log_frequency=10,
                              grad_reg_loss_type=args.grad_reg_loss_type, 
                              grad_reg_loss_params=grad_reg_loss_params, 
                              applied_hardware=args.applied_hardware, dummy_input=(1, 3, 224, 224),
                              ref_latency=args.reference_latency)
   trainer.train()
   trainer.export(args.arch_path)

The complete example code can be found :githublink:`here <examples/nas/oneshot/proxylessnas>`.

**Input arguments of ProxylessNasTrainer**


* **model** (*PyTorch model, required*\ ) - The model that users want to tune/search. It has mutables to specify search space.
* **metrics** (*PyTorch module, required*\ ) - The main term of the loss function for model train. Receives logits and ground truth label, return a loss tensor.
* **optimizer** (*PyTorch Optimizer, required*\) - The optimizer used for optimizing the model.
* **num_epochs** (*int, optional, default = 120*\ ) - The number of epochs to train/search.
* **dataset** (*PyTorch dataset, required*\ ) - Dataset for training. Will be split for training weights and architecture weights.
* **warmup_epochs** (*int, optional, default = 0*\ ) - The number of epochs to do during warmup.
* **batch_size** (*int, optional, default = 64*\ ) - Batch size.
* **workers** (*int, optional, default = 4*\ ) - Workers for data loading.
* **device** (*device, optional, default = 'cpu'*\ ) - The devices that users provide to do the train/search. The trainer applies data parallel on the model for users.
* **log_frequency** (*int, optional, default = None*\ ) - Step count per logging.
* **arc_learning_rate** (*float, optional, default = 1e-3*\ ) - The learning rate of the architecture parameters optimizer.
* **grad_reg_loss_type** (*'mul#log', 'add#linear', or None, optional, default = 'add#linear'*\ ) - Regularization type to add hardware related loss. The trainer will not apply loss regularization when grad_reg_loss_type is set as None.
* **grad_reg_loss_params** (*dict, optional, default = None*\ ) - Regularization params. 'alpha' and 'beta' is required when ``grad_reg_loss_type`` is 'mul#log', 'lambda' is required when ``grad_reg_loss_type`` is 'add#linear'.
* **applied_hardware** (*string, optional, default = None*\ ) - Applied hardware for to constraint the model's latency. Latency is predicted by Microsoft nn-Meter (https://github.com/microsoft/nn-Meter). 
* **dummy_input** (*tuple, optional, default = (1, 3, 224, 224)*\ ) - The dummy input shape when applied to the target hardware.
* **ref_latency** (*float, optional, default = 65.0*\ ) - Reference latency value in the applied hardware (ms).


Implementation
--------------

The implementation on NNI is based on the `offical implementation <https://github.com/mit-han-lab/ProxylessNAS>`__. The official implementation supports two training approaches: gradient descent and RL based. In our current implementation on NNI, gradient descent training approach is supported. The complete support of ProxylessNAS is ongoing.

The official implementation supports different targeted hardware, including 'mobile', 'cpu', 'gpu8', 'flops'.  In NNI repo, the hardware latency prediction is supported by `Microsoft nn-Meter <https://github.com/microsoft/nn-Meter>`__. nn-Meter is an accurate inference latency predictor for DNN models on diverse edge devices. nn-Meter support four hardwares up to now, including *'cortexA76cpu_tflite21'*, *'adreno640gpu_tflite21'*, *'adreno630gpu_tflite21'*, and *'myriadvpu_openvino2019r2'*. Users can find more information about nn-Meter on its website. More hardware will be supported in the future. Users could find more details about applying ``nn-Meter`` `here <./HardwareAwareNAS.rst>`__ .

Below we will describe implementation details. Like other one-shot NAS algorithms on NNI, ProxylessNAS is composed of two parts: *search space* and *training approach*. For users to flexibly define their own search space and use built-in ProxylessNAS training approach, we put the specified search space in :githublink:`example code <examples/nas/oneshot/proxylessnas>` using :githublink:`NNI NAS interface <nni/retiarii/oneshot/pytorch/proxyless>`.

.. image:: ../../img/proxylessnas.png
   :target: ../../img/proxylessnas.png
   :alt: 


ProxylessNAS training approach is composed of ProxylessLayerChoice and ProxylessNasTrainer. ProxylessLayerChoice instantiates MixedOp for each mutable (i.e., LayerChoice), and manage architecture weights in MixedOp. **For DataParallel**\ , architecture weights should be included in user model. Specifically, in ProxylessNAS implementation, we add MixedOp to the corresponding mutable (i.e., LayerChoice) as a member variable. The ProxylessLayerChoice class also exposes two member functions, i.e., ``resample``\ , ``finalize_grad``\ , for the trainer to control the training of architecture weights.

ProxylessNasMutator also implements the forward logic of the mutables (i.e., LayerChoice).

Reproduce Results
-----------------

To reproduce the result, we first run the search, we found that though it runs many epochs the chosen architecture converges at the first several epochs. This is probably induced by hyper-parameters or the implementation, we are working on it. 