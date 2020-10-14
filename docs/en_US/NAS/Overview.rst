
Neural Architecture Search (NAS) on NNI
=======================================

.. contents::

Overview
--------

Automatic neural architecture search is taking an increasingly important role in finding better models. Recent research has proved the feasibility of automatic NAS and has lead to models that beat many manually designed and tuned models. Some representative works are `NASNet <https://arxiv.org/abs/1707.07012>`_\ , `ENAS <https://arxiv.org/abs/1802.03268>`_\ , `DARTS <https://arxiv.org/abs/1806.09055>`_\ , `Network Morphism <https://arxiv.org/abs/1806.10282>`_\ , and `Evolution <https://arxiv.org/abs/1703.01041>`_. Further, new innovations keep emerging.

However, it takes a great effort to implement NAS algorithms, and it's hard to reuse the code base of existing algorithms for new ones. To facilitate NAS innovations (e.g., the design and implementation of new NAS models, the comparison of different NAS models side-by-side, etc.), an easy-to-use and flexible programming interface is crucial.

With this motivation, our ambition is to provide a unified architecture in NNI, accelerate innovations on NAS, and apply state-of-the-art algorithms to real-world problems faster.

With the unified interface, there are two different modes for architecture search. `One <#supported-one-shot-nas-algorithms>`_ is the so-called one-shot NAS, where a super-net is built based on a search space and one-shot training is used to generate a good-performing child model. `The other <#supported-classic-nas-algorithms>`_ is the traditional search-based approach, where each child model within the search space runs as an independent trial. We call it classic NAS.

NNI also provides dedicated `visualization tool <#nas-visualization>`_ for users to check the status of the neural architecture search process.

Supported Classic NAS Algorithms
--------------------------------

The procedure of classic NAS algorithms is similar to hyper-parameter tuning, users use ``nnictl`` to start experiments and each model runs as a trial. The difference is that search space file is automatically generated from user model (with search space in it) by running ``nnictl ss_gen``. The following table listed supported tuning algorihtms for classic NAS mode. More algorihtms will be supported in future release.

.. list-table::
   :header-rows: 1

   * - Name
     - Brief Introduction of Algorithm
   * - `Random Search <https://github.com/microsoft/nni/tree/master/examples/tuners/random_nas_tuner>`_
     - Randomly pick a model from search space
   * - `PPO Tuner <https://nni.readthedocs.io/en/latest/Tuner/BuiltinTuner.html#PPOTuner>`_
     - PPO Tuner is a Reinforcement Learning tuner based on PPO algorithm. `Reference Paper <https://arxiv.org/abs/1707.06347>`_


Please refer to `here <ClassicNas.md>`_ for the usage of classic NAS algorithms.

Supported One-shot NAS Algorithms
---------------------------------

NNI currently supports the one-shot NAS algorithms listed below and is adding more. Users can reproduce an algorithm or use it on their own dataset. We also encourage users to implement other algorithms with `NNI API <#use-nni-api>`_\ , to benefit more people.

.. list-table::
   :header-rows: 1

   * - Name
     - Brief Introduction of Algorithm
   * - `ENAS <https://nni.readthedocs.io/en/latest/NAS/ENAS.html>`_
     - `Efficient Neural Architecture Search via Parameter Sharing <https://arxiv.org/abs/1802.03268>`_. In ENAS, a controller learns to discover neural network architectures by searching for an optimal subgraph within a large computational graph. It uses parameter sharing between child models to achieve fast speed and excellent performance.
   * - `DARTS <https://nni.readthedocs.io/en/latest/NAS/DARTS.html>`_
     - `DARTS: Differentiable Architecture Search <https://arxiv.org/abs/1806.09055>`_ introduces a novel algorithm for differentiable network architecture search on bilevel optimization.
   * - `P-DARTS <https://nni.readthedocs.io/en/latest/NAS/PDARTS.html>`_
     - `Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation <https://arxiv.org/abs/1904.12760>`_ is based on DARTS. It introduces an efficient algorithm which allows the depth of searched architectures to grow gradually during the training procedure.
   * - `SPOS <https://nni.readthedocs.io/en/latest/NAS/SPOS.html>`_
     - `Single Path One-Shot Neural Architecture Search with Uniform Sampling <https://arxiv.org/abs/1904.00420>`_ constructs a simplified supernet trained with a uniform path sampling method and applies an evolutionary algorithm to efficiently search for the best-performing architectures.
   * - `CDARTS <https://nni.readthedocs.io/en/latest/NAS/CDARTS.html>`_
     - `Cyclic Differentiable Architecture Search <https://arxiv.org/abs/****>`_ builds a cyclic feedback mechanism between the search and evaluation networks. It introduces a cyclic differentiable architecture search framework which integrates the two networks into a unified architecture.
   * - `ProxylessNAS <https://nni.readthedocs.io/en/latest/NAS/Proxylessnas.html>`_
     - `ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware <https://arxiv.org/abs/1812.00332>`_. It removes proxy, directly learns the architectures for large-scale target tasks and target hardware platforms.
   * - `TextNAS <https://nni.readthedocs.io/en/latest/NAS/TextNAS.html>`_
     - `TextNAS: A Neural Architecture Search Space tailored for Text Representation <https://arxiv.org/pdf/1912.10729.pdf>`_. It is a neural architecture search algorithm tailored for text representation.


One-shot algorithms run **standalone without nnictl**. NNI supports both PyTorch and Tensorflow 2.X.

Here are some common dependencies to run the examples. PyTorch needs to be above 1.2 to use ``BoolTensor``.


* tensorboard
* PyTorch 1.2+
* git

Please refer to `here <NasGuide.md>`_ for the usage of one-shot NAS algorithms.

One-shot NAS can be visualized with our visualization tool. Learn more details `here <./Visualization.md>`_.

Search Space Zoo
----------------

NNI provides some predefined search space which can be easily reused. By stacking the extracted cells, user can quickly reproduce those NAS models.

Search Space Zoo contains the following NAS cells:


* `DartsCell <./SearchSpaceZoo.md#DartsCell>`_
* `ENAS micro <./SearchSpaceZoo.md#ENASMicroLayer>`_
* `ENAS macro <./SearchSpaceZoo.md#ENASMacroLayer>`_
* `NAS Bench 201 <./SearchSpaceZoo.md#nas-bench-201>`_

Using NNI API to Write Your Search Space
----------------------------------------

The programming interface of designing and searching a model is often demanded in two scenarios.


#. When designing a neural network, there may be multiple operation choices on a layer, sub-model, or connection, and it's undetermined which one or combination performs best. So, it needs an easy way to express the candidate layers or sub-models.
#. When applying NAS on a neural network, it needs a unified way to express the search space of architectures, so that it doesn't need to update trial code for different search algorithms.

For using NNI NAS, we suggest users to first go through `the tutorial of NAS API for building search space <./WriteSearchSpace.md>`_.

NAS Visualization
-----------------

To help users track the process and status of how the model is searched under specified search space, we developed a visualization tool. It visualizes search space as a super-net and shows importance of subnets and layers/operations, as well as how the importance changes along with the search process. Please refer to `the document of NAS visualization <./Visualization.md>`_ for how to use it.

Reference and Feedback
----------------------


* To `report a bug <https://github.com/microsoft/nni/issues/new?template=bug-report.md>`_ for this feature in GitHub;
* To `file a feature or improvement request <https://github.com/microsoft/nni/issues/new?template=enhancement.md>`_ for this feature in GitHub.
