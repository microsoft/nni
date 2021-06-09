Model Evaluators
================

NNI provides some commonly used model evaluators for users' convenience. If these model evaluators do not meet users' requirement, they can customize new model evaluators following the tutorial `here <./WriteTrainer.rst>`__.

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Classification
    :noindex:

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Regression
    :noindex:

Usage of Model Evaluator
------------------------

.. _serializer:

**serialize** is mainly used for serializing model training logic. It enables re-instantiation of model evaluator in another process or machine. Re-instantiation is necessary because most of time model and evaluator should be sent to training services. ``serialize`` is implemented by recording the initialization parameters of user instantiated evaluator.

The evaluator related APIs provided by Retiarii have already supported serialization, for example ``pl.Classification``, ``pl.DataLoader``, no need to apply ``serialize`` on them. In the following case users should use ``serialize`` API manually.

If the initialization parameters of the evaluator APIs (e.g., ``pl.Classification``, ``pl.DataLoader``) are not primitive types (e.g., ``int``, ``string``), they should be applied with  ``serialize``. If those parameters' initialization parameters are not primitive types, ``serialize`` should also be applied. In a word, ``serialize`` should be applied recursively if necessary.