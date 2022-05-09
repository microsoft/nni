Construct Model Space
=====================

NNI provides powerful (and multi-level) APIs for users to easily express model space (or search space).

* *Mutation Primitives*: high-level APIs (e.g., ValueChoice, LayerChoice) that are utilities to build blocks in search space. In most cases, mutation pritimives should be straightforward yet expressive enough. **We strongly recommend users to try them first,** and report issues if those APIs are not satisfying.
* *Hyper-module Library*: plug-and-play modules that are proved useful. They are usually well studied in research, and comes with pre-searched results. (For example, the optimal activation function in `AutoActivation <https://arxiv.org/abs/1710.05941>`__ is reported to be `Swish <https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html>`__).
* *Mutator*: for advanced users only. NNI provides interface to customize new mutators for expressing more complicated model spaces.

The following table summarizes all the APIs we have provided for constructing search space.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Category
     - Brief Description
   * - :class:`LayerChoice <nni.retiarii.nn.pytorch.LayerChoice>`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Select from some PyTorch modules
   * - :class:`InputChoice <nni.retiarii.nn.pytorch.InputChoice>`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Select from some inputs (tensors)
   * - :class:`ValueChoice <nni.retiarii.nn.pytorch.ValueChoice>`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Select from some candidate values
   * - :class:`Repeat <nni.retiarii.nn.pytorch.Repeat>`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Repeat a block by a variable number of times
   * - :class:`Cell <nni.retiarii.nn.pytorch.Cell>`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Cell structure popularly used in literature
   * - :class:`NasBench101Cell <nni.retiarii.nn.pytorch.NasBench101Cell>`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Cell structure (variant) proposed by NAS-Bench-101
   * - :class:`NasBench201Cell <nni.retiarii.nn.pytorch.NasBench201Cell>`
     - :ref:`Mutation Primitives <mutation-primitives>`
     - Cell structure (variant) proposed by NAS-Bench-201
   * - :class:`AutoActivation <nni.retiarii.nn.pytorch.AutoActivation>`
     - :ref:`Hyper-modules Library <hyper-modules>`
     - Searching for activation functions
   * - :class:`Mutator <nni.retiarii.Mutator>`
     - :doc:`Mutator <mutator>`
     - Flexible mutations on graphs. :doc:`See tutorial here <mutator>`
