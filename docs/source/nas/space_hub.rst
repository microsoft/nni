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
   * - :class:`~nni.retiarii.hub.pytorch.NasBench101`
     - Search space benchmarked by `NAS-Bench-101 <http://proceedings.mlr.press/v97/ying19a/ying19a.pdf>`__
   * - :class:`~nni.retiarii.hub.pytorch.NasBench201`
     - Search space benchmarked by `NAS-Bench-201 <https://arxiv.org/abs/2001.00326>`__
   * - :class:`~nni.retiarii.hub.pytorch.NASNet`
     - Proposed by `Learning Transferable Architectures for Scalable Image Recognition <https://arxiv.org/abs/1707.07012>`__
   * - :class:`~nni.retiarii.hub.pytorch.ENAS`
     - Proposed by `Efficient neural architecture search via parameter sharing <https://arxiv.org/abs/1802.03268>`__, subtly different from NASNet
   * - :class:`~nni.retiarii.hub.pytorch.AmoebaNet`
     - Proposed by
    `Regularized evolution for image classifier architecture search <https://arxiv.org/abs/1802.01548>`__, subtly different from NASNet
   * - :class:`~nni.retiarii.hub.pytorch.PNAS`
     - Proposed by
    `Progressive neural architecture search <https://arxiv.org/abs/1712.00559>`__, subtly different from NASNet
   * - :class:`~nni.retiarii.hub.pytorch.DARTS`
     - Proposed by `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__, most popularly used in evaluating one-shot algorithms
   * - :class:`~nni.retiarii.hub.pytorch.ProxylessNAS`
     - Proposed by `ProxylessNAS <https://arxiv.org/abs/1812.00332>`__, based on MobileNetV2.
   * - :class:`~nni.retiarii.hub.pytorch.MobileNetV3Space`
     - The largest space in `TuNAS <https://arxiv.org/abs/2008.06120>`__.
   * - :class:`~nni.retiarii.hub.pytorch.ShuffleNetSpace`
     - Based on ShuffleNetV2, proposed by `Single Path One-shot <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610528.pdf>`__
   * - :class:`~nni.retiarii.hub.pytorch.AutoformerSpace`
     - Based on ViT, proposed by `Autoformer <https://arxiv.org/abs/2107.00651>`__

Using model spaces from hub
---------------------------

Here is an example of how to use the model spaces from space hub.


