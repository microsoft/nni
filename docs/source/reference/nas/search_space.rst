NAS Reference for Search Space
==============================

.. _mutation-primitives:

Mutation Pritimives
-------------------

LayerChoice
^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.LayerChoice
   :members:


InputChoice
^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.InputChoice
   :members:

.. autoclass:: nni.retiarii.nn.pytorch.ChosenInputs
   :members:

ValueChoice
^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.ValueChoice
   :members:
   :inherited-members: Module

ModelParameterChoice
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.ModelParameterChoice
   :members:
   :inherited-members: Module

Repeat
^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.Repeat
   :members:

Cell
^^^^

.. autoclass:: nni.retiarii.nn.pytorch.Cell
   :members:

NasBench101Cell
^^^^^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.NasBench101Cell
   :members:

NasBench201Cell
^^^^^^^^^^^^^^^

.. autoclass:: nni.retiarii.nn.pytorch.NasBench201Cell
   :members:

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

Mutator
^^^^^^^

..  autoclass:: nni.retiarii.Mutator
    :members:

..  autoclass:: nni.retiarii.Sampler
    :members:

..  autoclass:: nni.retiarii.InvalidMutation
    :members:

Placeholder
^^^^^^^^^^^

..  autoclass:: nni.retiarii.nn.pytorch.Placeholder
    :members:

Graph
^^^^^

..  autoclass:: nni.retiarii.Model
    :members:

..  autoclass:: nni.retiarii.Graph
    :members:

..  autoclass:: nni.retiarii.Node
    :members:

..  autoclass:: nni.retiarii.Edge
    :members:

..  autoclass:: nni.retiarii.Operation
    :members:
