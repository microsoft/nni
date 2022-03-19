Construct Model Space
=====================

NNI provides powerful APIs for users to easily express model space (or search space).
Firstly, users can use high-level APIs (e.g., ValueChoice, LayerChoice) which are building blocks / skeletons of building blocks to construct their search space.
For advanced cases, NNI also provides interface to customize new mutators for expressing more complicated model spaces.

.. tip:: In most cases, this should be simple but expressive enough. We strongly recommend users to try them first, and report issues if those APIs are not satisfying.

.. _mutation-primitives:

Mutation Primitives
-------------------

To make users easily express a model space within their PyTorch/TensorFlow model, NNI provides some inline mutation APIs as shown below.

.. note:: We can actively adding more mutation primitives. If you have any suggestions, feel free to `ask here <https://github.com/microsoft/nni/issues>`__.

.. _nas-layer-choice:

LayerChoice
^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.LayerChoice
   :members:

.. _nas-input-choice:

InputChoice
^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.InputChoice
   :members:

.. autoclass:: nni.retiarii.nn.pytorch.ChosenInputs
   :members:

.. _nas-value-choice:

ValueChoice
^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.ValueChoice
   :members:
   :inherited-members: Module

.. _nas-repeat:

Repeat
^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.Repeat
   :members:

.. _nas-cell:

Cell
^^^^

.. autoclass:: nni.retiarii.nn.pytorch.Cell
   :members:

.. footbibliography::

.. _nas-cell-101:

NasBench101Cell
^^^^^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.NasBench101Cell
   :members:

.. footbibliography::

.. _nas-cell-201:

NasBench201Cell
^^^^^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.NasBench201Cell
   :members:

.. footbibliography::

.. _hyper-modules:

Hyper-module Library (experimental)
-----------------------------------

Hyper-module is a (PyTorch) module which contains many architecture/hyperparameter candidates for this module. By using hypermodule in user defined model, NNI will help users automatically find the best architecture/hyperparameter of the hyper-modules for this model. This follows the design philosophy of Retiarii that users write DNN model as a space.

We are planning to support some of the hyper-modules commonly used in the community, such as AutoDropout, AutoActivation. These are considered complementary to :ref:`mutation-primitives`, as they are often more concrete, specific, and tailored for particular needs.

.. _nas-autoactivation:

AutoActivation
^^^^^^^^^^^^^^

..  autoclass:: nni.retiarii.nn.pytorch.AutoActivation
    :members:

Mutators (advanced)
-------------------

Besides the inline mutation APIs demonstrated :ref:`above <mutation-primitives>`, NNI provides a more general approach to express a model space, i.e., *Mutator*, to cover more complex model spaces. Those inline mutation APIs are also implemented with mutator in the underlying system, which can be seen as a special case of model mutation. Please read :doc:`./mutator` for details.
