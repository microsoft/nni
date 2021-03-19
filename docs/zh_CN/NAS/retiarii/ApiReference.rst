Retiarii API 参考
======================

.. contents::

内联 Mutation API
----------------------------------------

..  autoclass:: nni.retiarii.nn.pytorch.LayerChoice
    :members:

..  autoclass:: nni.retiarii.nn.pytorch.InputChoice
    :members:

..  autoclass:: nni.retiarii.nn.pytorch.ValueChoice
    :members:

..  autoclass:: nni.retiarii.nn.pytorch.ChosenInputs
    :members:

图 Mutation API
--------------------------------------

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

Oneshot Trainers
----------------

..  autoclass:: nni.retiarii.oneshot.pytorch.DartsTrainer
    :members:

..  autoclass:: nni.retiarii.oneshot.pytorch.EnasTrainer
    :members:

..  autoclass:: nni.retiarii.oneshot.pytorch.ProxylessTrainer
    :members:

..  autoclass:: nni.retiarii.oneshot.pytorch.SinglePathTrainer
    :members:

Strategies
----------

..  autoclass:: nni.retiarii.strategy.Random
    :members:

..  autoclass:: nni.retiarii.strategy.GridSearch
    :members:

..  autoclass:: nni.retiarii.strategy.RegularizedEvolution
    :members:

..  autoclass:: nni.retiarii.strategy.TPEStrategy
    :members:

Retiarii Experiments
--------------------

..  autoclass:: nni.retiarii.experiment.pytorch.RetiariiExperiment
    :members:

..  autoclass:: nni.retiarii.experiment.pytorch.RetiariiExeConfig
    :members:
