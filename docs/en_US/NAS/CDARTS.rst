CDARTS
======

Introduction
------------

`CDARTS <https://arxiv.org/pdf/2006.10724.pdf>`__ builds a cyclic feedback mechanism between the search and evaluation networks. First, the search network generates an initial topology for evaluation, so that the weights of the evaluation network can be optimized. Second, the architecture topology in the search network is further optimized by the label supervision in classification, as well as the regularization from the evaluation network through feature distillation. Repeating the above cycle results in a joint optimization of the search and evaluation networks, and thus enables the evolution of the topology to fit the final evaluation network.

In implementation of ``CdartsTrainer``\ , it first instantiates two models and two mutators (one for each). The first model is the so-called "search network", which is mutated with a ``RegularizedDartsMutator`` -- a mutator with subtle differences with ``DartsMutator``. The second model is the "evaluation network", which is mutated with a discrete mutator that leverages the previous search network mutator, to sample a single path each time. Trainers train models and mutators alternatively. Users can refer to `paper <https://arxiv.org/pdf/2006.10724.pdf>`__ if they are interested in more details on these trainers and mutators.

Reproduction Results
--------------------

This is CDARTS based on the NNI platform, which currently supports CIFAR10 search and retrain. ImageNet search and retrain should also be supported, and we provide corresponding interfaces. Our reproduced results on NNI are slightly lower than the paper, but much higher than the original DARTS. Here we show the results of three independent experiments on CIFAR10.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Runs
     - Paper
     - NNI
   * - 1
     - 97.52
     - 97.44
   * - 2
     - 97.53
     - 97.48
   * - 3
     - 97.58
     - 97.56


Examples
--------

`Example code <https://github.com/microsoft/nni/tree/master/examples/nas/cdarts>`__

.. code-block:: bash

   # In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
   git clone https://github.com/Microsoft/nni.git

   # install apex for distributed training.
   git clone https://github.com/NVIDIA/apex
   cd apex
   python setup.py install --cpp_ext --cuda_ext

   # search the best architecture
   cd examples/nas/cdarts
   bash run_search_cifar.sh

   # train the best architecture.
   bash run_retrain_cifar.sh

Reference
---------

PyTorch
^^^^^^^

..  autoclass:: nni.algorithms.nas.pytorch.cdarts.CdartsTrainer
    :members:

..  autoclass:: nni.algorithms.nas.pytorch.cdarts.RegularizedDartsMutator
    :members:

..  autoclass:: nni.algorithms.nas.pytorch.cdarts.DartsDiscreteMutator
    :members:

..  autoclass:: nni.algorithms.nas.pytorch.cdarts.RegularizedMutatorParallel
    :members:
