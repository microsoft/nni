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

.. note:: The usage of one-shot has been refreshed in v2.8. Please see :doc:`legacy one-shot trainers </deprecated/oneshot_legacy>` for the old-style one-shot strategies.

DARTS
^^^^^

.. autoclass:: nni.retiarii.strategy.DARTS
   :members:

ENAS
^^^^^

.. autoclass:: nni.retiarii.strategy.ENAS
   :members:

.. autoclass:: nni.retiarii.oneshot.pytorch.enas.ReinforceController
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

base_lightning
""""""""""""""

..  automodule:: nni.retiarii.oneshot.pytorch.base_lightning
    :members:

dataloader
""""""""""

..  automodule:: nni.retiarii.oneshot.pytorch.dataloader
    :members:

supermodule.differentiable
""""""""""""""""""""""""""

..  automodule:: nni.retiarii.oneshot.pytorch.supermodule.differentiable
    :members:

supermodule.sampling
""""""""""""""""""""

..  automodule:: nni.retiarii.oneshot.pytorch.supermodule.sampling
    :members:

supermodule.proxyless
"""""""""""""""""""""

..  automodule:: nni.retiarii.oneshot.pytorch.supermodule.proxyless
    :members:

supermodule.operation
"""""""""""""""""""""

..  automodule:: nni.retiarii.oneshot.pytorch.supermodule.operation
    :members:
