Search Space
============

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

AutoActivation
^^^^^^^^^^^^^^

..  autoclass:: nni.retiarii.nn.pytorch.AutoActivation
    :members:

Model Space Hub
---------------

NasBench101
^^^^^^^^^^^

..  autoclass:: nni.retiarii.hub.pytorch.NasBench101
    :members:

NasBench201
^^^^^^^^^^^

..  autoclass:: nni.retiarii.hub.pytorch.NasBench201
    :members:

NASNet
^^^^^^

..  autoclass:: nni.retiarii.hub.pytorch.NASNet
    :members:

..  autoclass:: nni.retiarii.hub.pytorch.nasnet.NDS
    :members:

..  autoclass:: nni.retiarii.hub.pytorch.nasnet.NDSStage
    :members:

..  autoclass:: nni.retiarii.hub.pytorch.nasnet.NDSStagePathSampling
    :members:

..  autoclass:: nni.retiarii.hub.pytorch.nasnet.NDSStageDifferentiable
    :members:

ENAS
^^^^

..  autoclass:: nni.retiarii.hub.pytorch.ENAS
    :members:

AmoebaNet
^^^^^^^^^

..  autoclass:: nni.retiarii.hub.pytorch.AmoebaNet
    :members:

PNAS
^^^^

..  autoclass:: nni.retiarii.hub.pytorch.PNAS
    :members:

DARTS
^^^^^

..  autoclass:: nni.retiarii.hub.pytorch.DARTS
    :members:

ProxylessNAS
^^^^^^^^^^^^

..  autoclass:: nni.retiarii.hub.pytorch.ProxylessNAS
    :members:

..  autoclass:: nni.retiarii.hub.pytorch.proxylessnas.InvertedResidual
    :members:

MobileNetV3Space
^^^^^^^^^^^^^^^^

..  autoclass:: nni.retiarii.hub.pytorch.MobileNetV3Space
    :members:

ShuffleNetSpace
^^^^^^^^^^^^^^^

..  autoclass:: nni.retiarii.hub.pytorch.ShuffleNetSpace
    :members:

AutoformerSpace
^^^^^^^^^^^^^^^

..  autoclass:: nni.retiarii.hub.pytorch.AutoformerSpace
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
