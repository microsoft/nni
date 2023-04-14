Model Space Hub
===============

NNI model space hub contains a curated list of well-known NAS search spaces, along with a number of famous model space building blocks. Consider reading this document or try the models / spaces provided in the hub if you intend to:

1. Use a pre-defined model space as a starting point for your model development.
2. Try the state-of-the-art searched architecture along with its associated weights in your own task.
3. Learn the performance of NNI's built-in NAS search strategies on some well-recognized model spaces.
4. Build and test your NAS algorithm on the space hub and fairly compare them with other baselines.

List of supported model spaces
------------------------------

The model spaces provided so far are all built for image classification tasks, though they can serve as backbones for downstream tasks.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Description
   * - :class:`~nni.nas.hub.pytorch.NasBench101`
     - Search space benchmarked by `NAS-Bench-101 <http://proceedings.mlr.press/v97/ying19a/ying19a.pdf>`__
   * - :class:`~nni.nas.hub.pytorch.NasBench201`
     - Search space benchmarked by `NAS-Bench-201 <https://arxiv.org/abs/2001.00326>`__
   * - :class:`~nni.nas.hub.pytorch.NASNet`
     - Proposed by `Learning Transferable Architectures for Scalable Image Recognition <https://arxiv.org/abs/1707.07012>`__
   * - :class:`~nni.nas.hub.pytorch.ENAS`
     - Proposed by `Efficient neural architecture search via parameter sharing <https://arxiv.org/abs/1802.03268>`__, subtly different from NASNet
   * - :class:`~nni.nas.hub.pytorch.AmoebaNet`
     - Proposed by `Regularized evolution for image classifier architecture search <https://arxiv.org/abs/1802.01548>`__, subtly different from NASNet
   * - :class:`~nni.nas.hub.pytorch.PNAS`
     - Proposed by `Progressive neural architecture search <https://arxiv.org/abs/1712.00559>`__, subtly different from NASNet
   * - :class:`~nni.nas.hub.pytorch.DARTS`
     - Proposed by `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__, most popularly used in evaluating one-shot algorithms
   * - :class:`~nni.nas.hub.pytorch.ProxylessNAS`
     - Proposed by `ProxylessNAS <https://arxiv.org/abs/1812.00332>`__, based on MobileNetV2.
   * - :class:`~nni.nas.hub.pytorch.MobileNetV3Space`
     - The largest space in `TuNAS <https://arxiv.org/abs/2008.06120>`__.
   * - :class:`~nni.nas.hub.pytorch.ShuffleNetSpace`
     - Based on ShuffleNetV2, proposed by `Single Path One-shot <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610528.pdf>`__
   * - :class:`~nni.nas.hub.pytorch.AutoFormer`
     - Based on ViT, proposed by `Autoformer <https://arxiv.org/abs/2107.00651>`__

.. note::

   We are actively enriching the model space hub. Planned model spaces include:

   - `NAS-BERT <https://arxiv.org/abs/2105.14444>`__
   - `LightSpeech <https://arxiv.org/abs/2102.04040>`__

   We welcome suggestions and contributions.

Using pre-searched models
-------------------------

One way to use the model space is to directly leverage the searched results. Note that some of them have already been well-known neural networks and widely used.

.. code-block:: python

   import torch
   from nni.nas.hub.pytorch import MobileNetV3Space
   from torch.utils.data import DataLoader
   from torchvision import transforms
   from torchvision.datasets import ImageNet

   # Load one of the searched results from MobileNetV3 search space.
   mobilenetv3 = MobileNetV3Space.load_searched_model(
       'mobilenetv3-small-100',        # Available model alias are listed in the table below.
       pretrained=True, download=True  # download and load the pretrained checkpoint
   )

   # MobileNetV3 model can be directly evaluated on ImageNet
   transform = transforms.Compose([
       transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   dataset = ImageNet('/path/to/your/imagenet', 'val', transform=transform)
   dataloader = DataLoader(dataset, batch_size=64)
   mobilenetv3.eval()
   with torch.no_grad():
       correct = total = 0
       for inputs, targets in dataloader:
           logits = mobilenetv3(inputs)
           _, predict = torch.max(logits, 1)
           correct += (predict == targets).sum().item()
           total += targets.size(0)
   print('Accuracy:', correct / total)

In the example above, ``MobileNetV3Space`` can be replaced with any model spaces in the hub, and ``mobilenetv3-small-100`` can be any model alias listed below.

+-------------------+------------------------+----------+---------+-------------------------------+
| Search space      | Model                  | Dataset  | Metric  | Eval configurations           |
+===================+========================+==========+=========+===============================+
| ProxylessNAS      | acenas-m1              | ImageNet | 75.176  | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| ProxylessNAS      | acenas-m2              | ImageNet | 75.0    | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| ProxylessNAS      | acenas-m3              | ImageNet | 75.118  | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| ProxylessNAS      | proxyless-cpu          | ImageNet | 75.29   | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| ProxylessNAS      | proxyless-gpu          | ImageNet | 75.084  | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| ProxylessNAS      | proxyless-mobile       | ImageNet | 74.594  | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | mobilenetv3-large-100  | ImageNet | 75.768  | Bicubic interpolation         |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | mobilenetv3-small-050  | ImageNet | 57.906  | Bicubic interpolation         |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | mobilenetv3-small-075  | ImageNet | 65.24   | Bicubic interpolation         |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | mobilenetv3-small-100  | ImageNet | 67.652  | Bicubic interpolation         |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | cream-014              | ImageNet | 53.74   | Test image size = 64          |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | cream-043              | ImageNet | 66.256  | Test image size = 96          |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | cream-114              | ImageNet | 72.514  | Test image size = 160         |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | cream-287              | ImageNet | 77.52   | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | cream-481              | ImageNet | 79.078  | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| MobileNetV3Space  | cream-604              | ImageNet | 79.92   | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| DARTS             | darts-v2               | CIFAR-10 | 97.37   | Default                       |
+-------------------+------------------------+----------+---------+-------------------------------+
| ShuffleNetSpace   | spos                   | ImageNet | 74.14   | BGR tensor; no normalization  |
+-------------------+------------------------+----------+---------+-------------------------------+

.. note::

   1. The metrics listed above are obtained by evaluating the checkpoints provided by the original author and converted to NNI NAS format with `these scripts <https://github.com/ultmaster/spacehub-conversion>`__. Do note that some metrics can be higher / lower than the original report, because there could be subtle differences between data preprocessing, operation implementation (e.g., 3rd-party hswish vs ``nn.Hardswish``), or even library versions we are using. But most of these errors are acceptable (~0.1%).
   2. The default metric for ImageNet and CIFAR-10 is top-1 accuracy.
   3. Refer to `timm <https://github.com/rwightman/pytorch-image-models>`__ for the evaluation configurations.

.. todos: measure latencies and flops, reproduce training.

Searching within model spaces
-----------------------------

To search within a model space for a new architecture on a particular dataset,
users need to create model space, search strategy, and evaluator following the :doc:`standard procedures </tutorials/hello_nas>`.

Here is a short sample code snippet for reference.

.. code-block:: python

   # Create the model space
   from nni.nas.hub.pytorch import MobileNetV3Space
   model_space = MobileNetV3Space()

   # Pick a search strategy
   from nni.nas.strategy import RegularizedEvolution
   strategy = RegularizedEvolution()  # It can be any strategy, including one-shot strategies.

   # Define an evaluator
   from nni.nas.evaluator.pytorch import Classification
   evaluator = Classification(train_dataloaders=DataLoader(train_dataset, batch_size=batch_size),
                              val_dataloaders=DataLoader(test_dataset, batch_size=batch_size))

   # Launch the experiment, start the search process
   experiment = NasExperiment(model_space, evaluator, strategy)
   experiment.run()

.. todo: search reproduction results
