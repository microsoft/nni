.. role:: raw-html(raw)
   :format: html


Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search
=======================================================================================

`[Paper] <https://papers.nips.cc/paper/2020/file/d072677d210ac4c03ba046120f0802ec-Paper.pdf>`__ `[Models-Google Drive] <https://drive.google.com/drive/folders/1NLGAbBF9bA1IUAxKlk2VjgRXhr6RHvRW?usp=sharing>`__ `[Models-Baidu Disk (PWD: wqw6)] <https://pan.baidu.com/s/1TqQNm2s14oEdyNPimw3T9g>`__ `[BibTex] <https://scholar.googleusercontent.com/scholar.bib?q=info:ICWVXc_SsKAJ:scholar.google.com/&output=citation&scisdr=CgUmooXfEMfTi0cV5aU:AAGBfm0AAAAAX7sQ_aXoamdKRaBI12tAVN8REq1VKNwM&scisig=AAGBfm0AAAAAX7sQ_RdYtp6BSro3zgbXVJU2MCgsG730&scisf=4&ct=citation&cd=-1&hl=ja>`__   :raw-html:`<br/>`

In this work, we present a simple yet effective architecture distillation method. The central idea is that subnetworks can learn collaboratively and teach each other throughout the training process, aiming to boost the convergence of individual models. We introduce the concept of prioritized path, which refers to the architecture candidates exhibiting superior performance during training. Distilling knowledge from the prioritized paths is able to boost the training of subnetworks. Since the prioritized paths are changed on the fly depending on their performance and complexity, the final obtained paths are the cream of the crop. The discovered architectures achieve superior performance compared to the recent `MobileNetV3 <https://arxiv.org/abs/1905.02244>`__ and `EfficientNet <https://arxiv.org/abs/1905.11946>`__ families under aligned settings.

:raw-html:`<div ><img src="https://github.com/microsoft/Cream/blob/main/demo/intro.jpg" width="800"/></div>`
Reproduced Results
------------------

Top-1 Accuracy on ImageNet. The top-1 accuracy of Cream search algorithm surpasses MobileNetV3 and EfficientNet-B0/B1 on ImageNet.
The training with 16 Gpus is a little bit superior than 8 Gpus, as below.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Model (M Flops)
     - 8Gpus
     - 16Gpus
   * - 14M
     - 53.7
     - 53.8
   * - 43M
     - 65.8
     - 66.5
   * - 114M
     - 72.1
     - 72.8
   * - 287M
     - 76.7
     - 77.6
   * - 481M
     - 78.9
     - 79.2
   * - 604M
     - 79.4
     - 80.0



.. raw:: html

   <table style="border: none">
       <th><img src="./../../img/cream_flops100.jpg" alt="drawing" width="400"/></th>
       <th><img src="./../../img/cream_flops600.jpg" alt="drawing" width="400"/></th>
   </table>


Examples
--------

`Example code <https://github.com/microsoft/nni/tree/master/examples/nas/cream>`__

Please run the following scripts in the example folder.

Data Preparation
----------------

You need to first download the `ImageNet-2012 <http://www.image-net.org/>`__ to the folder ``./data/imagenet`` and move the validation set to the subfolder ``./data/imagenet/val``. To move the validation set, you cloud use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh 

Put the imagenet data in ``./data``. It should be like following:

.. code-block:: bash

   ./data/imagenet/train
   ./data/imagenet/val
   ...

Quick Start
-----------

I. Search
^^^^^^^^^

First, build environments for searching.

.. code-block:: bash

   pip install -r ./requirements

   git clone https://github.com/NVIDIA/apex.git
   cd apex
   python setup.py install --cpp_ext --cuda_ext

To search for an architecture, you need to configure the parameters ``FLOPS_MINIMUM`` and ``FLOPS_MAXIMUM`` to specify the desired model flops, such as [0,600]MB flops. You can specify the flops interval by changing these two parameters in ``./configs/train.yaml``

.. code-block:: bash

   FLOPS_MINIMUM: 0 # Minimum Flops of Architecture
   FLOPS_MAXIMUM: 600 # Maximum Flops of Architecture

For example, if you expect to search an architecture with model flops <= 200M, please set the ``FLOPS_MINIMUM`` and ``FLOPS_MAXIMUM`` to be ``0`` and ``200``.

After you specify the flops of the architectures you would like to search, you can search an architecture now by running:

.. code-block:: bash

   python -m torch.distributed.launch --nproc_per_node=8 ./train.py --cfg ./configs/train.yaml

The searched architectures need to be retrained and obtain the final model. The final model is saved in ``.pth.tar`` format. Retraining code will be released soon.

II. Retrain
^^^^^^^^^^^

To train searched architectures, you need to configure the parameter ``MODEL_SELECTION`` to specify the model Flops. To specify which model to train, you should add ``MODEL_SELECTION`` in ``./configs/retrain.yaml``. You can select one from [14,43,112,287,481,604], which stands for different Flops(MB).

.. code-block:: bash

   MODEL_SELECTION: 43 # Retrain 43m model
   MODEL_SELECTION: 481 # Retrain 481m model
   ......

To train random architectures, you need specify ``MODEL_SELECTION`` to ``-1`` and configure the parameter ``INPUT_ARCH``\ :

.. code-block:: bash

   MODEL_SELECTION: -1 # Train random architectures
   INPUT_ARCH: [[0], [3], [3, 3], [3, 1, 3], [3, 3, 3, 3], [3, 3, 3], [0]] # Random Architectures
   ......

After adding ``MODEL_SELECTION`` in ``./configs/retrain.yaml``\ , you need to use the following command to train the model.

.. code-block:: bash

   python -m torch.distributed.launch --nproc_per_node=8 ./retrain.py --cfg ./configs/retrain.yaml

III. Test
^^^^^^^^^

To test our trained of models, you need to use ``MODEL_SELECTION`` in ``./configs/test.yaml`` to specify which model to test.

.. code-block:: bash

   MODEL_SELECTION: 43 # test 43m model
   MODEL_SELECTION: 481 # test 470m model
   ......

After specifying the flops of the model, you need to write the path to the resume model in ``./test.sh``.

.. code-block:: bash

   RESUME_PATH: './43.pth.tar'
   RESUME_PATH: './481.pth.tar'
   ......

We provide 14M/43M/114M/287M/481M/604M pretrained models in `google drive <https://drive.google.com/drive/folders/1CQjyBryZ4F20Rutj7coF8HWFcedApUn2>`__ or `[Models-Baidu Disk (password: wqw6)] <https://pan.baidu.com/s/1TqQNm2s14oEdyNPimw3T9g>`__ .

After downloading the pretrained models and adding ``MODEL_SELECTION`` and ``RESUME_PATH`` in './configs/test.yaml', you need to use the following command to test the model.

.. code-block:: bash

   python -m torch.distributed.launch --nproc_per_node=8 ./test.py --cfg ./configs/test.yaml
