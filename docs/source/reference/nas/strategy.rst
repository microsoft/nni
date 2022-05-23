Strategy
========

.. _multi-trial-nas-reference:

Multi-trial Strategy
--------------------

Random
^^^^^^

.. autoclass:: nni.retiarii.strategy.Random
   :members:

GridSearch
^^^^^^^^^^

.. autoclass:: nni.retiarii.strategy.GridSearch
   :members:

RegularizedEvolution
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: nni.retiarii.strategy.RegularizedEvolution
   :members:

TPE
^^^

.. autoclass:: nni.retiarii.strategy.TPE
   :members:

PolicyBasedRL
^^^^^^^^^^^^^

.. autoclass:: nni.retiarii.strategy.PolicyBasedRL
   :members:

.. _one-shot-strategy-reference:

One-shot Strategy
-----------------

.. note:: The usage of one-shot has been refreshed in v2.8. Please see :doc:`legacy one-shot trainers <oneshot_legacy>` for the old-style one-shot strategies.

DARTS
^^^^^

.. autoclass:: nni.retiarii.strategy.DARTS
   :members:

ENAS
^^^^^

.. autoclass:: nni.retiarii.strategy.ENAS
   :members:

GumbelDARTS
^^^^^^^^^^^

.. autoclass:: nni.retiarii.strategy.GumbelDARTS
   :members:

RandomOneShot
^^^^^^^^^^^^^

.. autoclass:: nni.retiarii.strategy.RandomOneShot
   :members:

Proxyless
^^^^^^^^^

.. autoclass:: nni.retiarii.strategy.Proxyless
   :members:


Customization
-------------

Multi-trial
^^^^^^^^^^^

..  autoclass:: nni.retiarii.Sampler
    :noindex:
    :members:

..  autoclass:: nni.retiarii.strategy.BaseStrategy
    :members:

..  automodule:: nni.retiarii.execution
    :members:
    :imported-members:
    :undoc-members:

One-shot
^^^^^^^^

..  automodule:: nni.retiarii.oneshot.pytorch.base_lightning
    :members:

..  automodule:: nni.retiarii.oneshot.pytorch.supermodule.differentiable
    :members:

..  automodule:: nni.retiarii.oneshot.pytorch.supermodule.sampling
    :members:

..  automodule:: nni.retiarii.oneshot.pytorch.supermodule.proxyless
    :members:

..  automodule:: nni.retiarii.oneshot.pytorch.supermodule.operation
    :members:

..  automodule:: nni.retiarii.oneshot.pytorch.dataloader
    :members:
