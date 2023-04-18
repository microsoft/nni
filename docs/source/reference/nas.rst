NAS API Reference
=================

Model space
-----------

.. autoclass:: nni.nas.nn.pytorch.LayerChoice
   :members:

.. autoclass:: nni.nas.nn.pytorch.InputChoice
   :members:

.. autoclass:: nni.nas.nn.pytorch.Repeat
   :members:

.. autoclass:: nni.nas.nn.pytorch.Cell
   :members:

.. autoclass:: nni.nas.nn.pytorch.ModelSpace
   :members:

.. autoclass:: nni.nas.nn.pytorch.ParametrizedModule
   :members:

.. autoclass:: nni.nas.nn.pytorch.MutableModule
   :members:

Model Space Hub
---------------

NasBench101
^^^^^^^^^^^

..  autoclass:: nni.nas.hub.pytorch.NasBench101
    :members:

NasBench201
^^^^^^^^^^^

..  autoclass:: nni.nas.hub.pytorch.NasBench201
    :members:

NASNet
^^^^^^

..  autoclass:: nni.nas.hub.pytorch.NASNet
    :members:

..  autoclass:: nni.nas.hub.pytorch.nasnet.NDS
    :members:

..  autoclass:: nni.nas.hub.pytorch.nasnet.NDSStage
    :members:

..  autoclass:: nni.nas.hub.pytorch.nasnet.NDSStagePathSampling
    :members:

..  autoclass:: nni.nas.hub.pytorch.nasnet.NDSStageDifferentiable
    :members:

ENAS
^^^^

..  autoclass:: nni.nas.hub.pytorch.ENAS
    :members:

AmoebaNet
^^^^^^^^^

..  autoclass:: nni.nas.hub.pytorch.AmoebaNet
    :members:

PNAS
^^^^

..  autoclass:: nni.nas.hub.pytorch.PNAS
    :members:

DARTS
^^^^^

..  autoclass:: nni.nas.hub.pytorch.DARTS
    :members:

ProxylessNAS
^^^^^^^^^^^^

..  autoclass:: nni.nas.hub.pytorch.ProxylessNAS
    :members:

..  autoclass:: nni.nas.hub.pytorch.proxylessnas.InvertedResidual
    :members:

MobileNetV3Space
^^^^^^^^^^^^^^^^

..  autoclass:: nni.nas.hub.pytorch.MobileNetV3Space
    :members:

ShuffleNetSpace
^^^^^^^^^^^^^^^

..  autoclass:: nni.nas.hub.pytorch.ShuffleNetSpace
    :members:

AutoFormer
^^^^^^^^^^

..  autoclass:: nni.nas.hub.pytorch.AutoFormer
    :members:

Module Components
^^^^^^^^^^^^^^^^^

.. automodule:: nni.nas.hub.pytorch.modules
   :members:
   :imported-members:

Evaluator
---------

.. autoclass:: nni.nas.evaluator.FunctionalEvaluator
   :members:

.. autoclass:: nni.nas.evaluator.Evaluator
   :members:

.. autoclass:: nni.nas.evaluator.MutableEvaluator
   :members:

..  autoclass:: nni.nas.evaluator.pytorch.Classification
    :members:

..  autoclass:: nni.nas.evaluator.pytorch.ClassificationModule
    :members:

..  autoclass:: nni.nas.evaluator.pytorch.Regression
    :members:

..  autoclass:: nni.nas.evaluator.pytorch.RegressionModule
    :members:

..  autoclass:: nni.nas.evaluator.pytorch.Trainer

..  autoclass:: nni.nas.evaluator.pytorch.DataLoader

..  autoclass:: nni.nas.evaluator.pytorch.Lightning
    :members:

..  autoclass:: nni.nas.evaluator.pytorch.LightningModule
    :members:

Multi-trial strategy
--------------------

.. autoclass:: nni.nas.strategy.GridSearch
   :members:

.. autoclass:: nni.nas.strategy.Random
   :members:

.. autoclass:: nni.nas.strategy.RegularizedEvolution
   :members:

.. autoclass:: nni.nas.strategy.PolicyBasedRL
   :members:

.. autoclass:: nni.nas.strategy.TPE
   :members:

Advanced APIs
^^^^^^^^^^^^^

Base
""""

.. automodule:: nni.nas.strategy.base
   :members:

Middleware
""""""""""

.. automodule:: nni.nas.strategy.middleware
   :members:

Utilities
"""""""""

.. automodule:: nni.nas.strategy.utils
   :members:

One-shot strategies
-------------------

.. autoclass:: nni.nas.strategy.RandomOneShot
   :members:

.. autoclass:: nni.nas.strategy.ENAS
   :members:

.. autoclass:: nni.nas.strategy.DARTS
   :members:

.. autoclass:: nni.nas.strategy.GumbelDARTS
   :members:

.. autoclass:: nni.nas.strategy.Proxyless
   :members:

Advanced APIs
^^^^^^^^^^^^^

.. autoclass:: nni.nas.oneshot.pytorch.strategy.OneShotStrategy
   :members:

base_lightning
""""""""""""""

..  automodule:: nni.nas.oneshot.pytorch.base_lightning
    :members:
    :imported-members:

supermodule.differentiable
""""""""""""""""""""""""""

..  automodule:: nni.nas.oneshot.pytorch.supermodule.differentiable
    :members:
    :imported-members:

supermodule.sampling
""""""""""""""""""""

..  automodule:: nni.nas.oneshot.pytorch.supermodule.sampling
    :members:
    :imported-members:

supermodule.proxyless
"""""""""""""""""""""

..  automodule:: nni.nas.oneshot.pytorch.supermodule.proxyless
    :members:
    :imported-members:

supermodule.operation
"""""""""""""""""""""

..  automodule:: nni.nas.oneshot.pytorch.supermodule.operation
    :members:
    :imported-members:

Profiler Utilities
""""""""""""""""""

.. automodule:: nni.nas.oneshot.pytorch.profiler
   :members:

Experiment
----------

.. autoclass:: nni.nas.experiment.NasExperiment
   :members:

.. automodule:: nni.nas.experiment.config
   :members:
   :imported-members:

Profiler
--------

.. autoclass:: nni.nas.profiler.Profiler
   :members:

.. autoclass:: nni.nas.profiler.ExpressionProfiler
   :members:

FLOPs
^^^^^

.. automodule:: nni.nas.profiler.pytorch.flops
   :members:

nn-Meter
^^^^^^^^

.. automodule:: nni.nas.profiler.pytorch.nn_meter
   :members:

Model format
------------

.. automodule:: nni.nas.space
   :members:
   :imported-members:

Execution engine
----------------

.. automodule:: nni.nas.execution
   :members:
   :imported-members:

Cross-graph optimization
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: nni.nas.execution.cgo
   :members:
   :imported-members:

NAS Benchmarks
--------------

.. automodule:: nni.nas.benchmark
   :members:
   :imported-members:

NAS-Bench-101
^^^^^^^^^^^^^

.. automodule:: nni.nas.benchmark.nasbench101
   :members:
   :imported-members:

NAS-Bench-201
^^^^^^^^^^^^^

.. automodule:: nni.nas.benchmark.nasbench201
   :members:
   :imported-members:

NDS
^^^

.. automodule:: nni.nas.benchmark.nds
   :members:
   :imported-members:

Miscellaneous Utilities
-----------------------

.. automodule:: nni.nas.utils.serializer
   :members:
