Retiarii API Reference
======================

.. contents::

Inline Mutation APIs
--------------------

..  autoclass:: nni.retiarii.nn.pytorch.LayerChoice
    :members:

..  autoclass:: nni.retiarii.nn.pytorch.InputChoice
    :members:

..  autoclass:: nni.retiarii.nn.pytorch.ValueChoice
    :members:

..  autoclass:: nni.retiarii.nn.pytorch.ChosenInputs
    :members:

..  autoclass:: nni.retiarii.nn.pytorch.Repeat
    :members:

..  autoclass:: nni.retiarii.nn.pytorch.Cell
    :members:

Graph Mutation APIs
-------------------

..  autoclass:: nni.retiarii.Mutator
    :members:

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

Evaluators
----------

..  autoclass:: nni.retiarii.evaluator.FunctionalEvaluator
    :members:

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.LightningModule
    :members:

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Classification
    :members:

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Regression
    :members:

Exploration Strategies
----------------------

..  automodule:: nni.retiarii.strategy
    :members:
    :imported-members:

Retiarii Experiments
--------------------

..  autoclass:: nni.retiarii.experiment.pytorch.RetiariiExperiment
    :members:

..  autoclass:: nni.retiarii.experiment.pytorch.RetiariiExeConfig
    :members:

CGO Execution
-------------

..  autofunction:: nni.retiarii.evaluator.pytorch.cgo.evaluator.MultiModelSupervisedLearningModule

..  autofunction:: nni.retiarii.evaluator.pytorch.cgo.evaluator.Classification

..  autofunction:: nni.retiarii.evaluator.pytorch.cgo.evaluator.Regression

One-shot Implementation
-----------------------

..  automodule:: nni.retiarii.oneshot
    :members:
    :imported-members:

..  automodule:: nni.retiarii.oneshot.pytorch
    :members:
    :imported-members:

Utilities
---------

..  autofunction:: nni.retiarii.basic_unit

..  autofunction:: nni.retiarii.model_wrapper

..  autofunction:: nni.retiarii.fixed_arch

Citations
---------

.. bibliography::
