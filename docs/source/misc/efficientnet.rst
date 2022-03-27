EfficientNet
============

`EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`__

Use Grid search to find the best combination of alpha, beta and gamma for EfficientNet-B1, as discussed in Section 3.3 in paper. Search space, tuner, configuration examples are provided here.

Instructions
------------

:githublink:`Example code <examples/trials/efficientnet>`


#. Set your working directory here in the example code directory.
#. Run ``git clone https://github.com/ultmaster/EfficientNet-PyTorch`` to clone the `ultmaster modified version <https://github.com/ultmaster/EfficientNet-PyTorch>`__ of the original `EfficientNet-PyTorch <https://github.com/lukemelas/EfficientNet-PyTorch>`__. The modifications were done to adhere to the original `Tensorflow version <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`__ as close as possible (including EMA, label smoothing and etc.); also added are the part which gets parameters from tuner and reports intermediate/final results. Clone it into ``EfficientNet-PyTorch``\ ; the files like ``main.py``\ , ``train_imagenet.sh`` will appear inside, as specified in the configuration files.
#. Run ``nnictl create --config config_local.yml`` (use ``config_pai.yml`` for OpenPAI) to find the best EfficientNet-B1. Adjust the training service (PAI/local/remote), batch size in the config files according to the environment.

For training on ImageNet, read ``EfficientNet-PyTorch/train_imagenet.sh``. Download ImageNet beforehand and extract it adhering to `PyTorch format <https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet>`__ and then replace ``/mnt/data/imagenet`` in with the location of the ImageNet storage. This file should also be a good example to follow for mounting ImageNet into the container on OpenPAI.

Results
-------

The follow image is a screenshot, demonstrating the relationship between acc@1 and alpha, beta, gamma.


.. image:: ../../img/efficientnet_search_result.png
   :target: ../../img/efficientnet_search_result.png
   :alt: 

