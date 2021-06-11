Model Evaluators
================

A model evaluator is for training and validating each generated model.

Usage of Model Evaluator
------------------------

In multi-trial NAS, a sampled model should be able to be executed on a remote machine or a training platform (e.g., AzureML, OpenPAI). Thus, both the model and its model evaluator should be correctly serialized. To make NNI correctly serialize model evaluator, users should apply ``serialize`` on some of their functions and objects.

.. _serializer:

`serialize <./ApiReference.rst#utilities>`__ enables re-instantiation of model evaluator in another process or machine. It is implemented by recording the initialization parameters of user instantiated evaluator.

The evaluator related APIs provided by Retiarii have already supported serialization, for example ``pl.Classification``, ``pl.DataLoader``, no need to apply ``serialize`` on them. In the following case users should use ``serialize`` API manually.

If the initialization parameters of the evaluator APIs (e.g., ``pl.Classification``, ``pl.DataLoader``) are not primitive types (e.g., ``int``, ``string``), they should be applied with  ``serialize``. If those parameters' initialization parameters are not primitive types, ``serialize`` should also be applied. In a word, ``serialize`` should be applied recursively if necessary.

Below is an example, ``transforms.Compose``, ``transforms.Normalize``, and ``MNIST`` are serialized manually using ``serialize``. ``serialize`` takes a class ``cls`` as its first argument, its following arguments are the arguments for initializing this class. ``pl.Classification`` is not applied ``serialize`` because it is already serializable as an API provided by NNI.

.. code-block:: python

  import nni.retiarii.evaluator.pytorch.lightning as pl
  from nni.retiarii import serialize
  from torchvision import transforms

  transform = serialize(transforms.Compose, [serialize(transforms.ToTensor()), serialize(transforms.Normalize, (0.1307,), (0.3081,))])
  train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)
  evaluator = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=10)

Supported Model Evaluators
--------------------------

NNI provides some commonly used model evaluators for users' convenience. If these model evaluators do not meet users' requirement, they can customize new model evaluators following the tutorial `here <./WriteTrainer.rst>`__.

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Classification
    :noindex:

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Regression
    :noindex:
