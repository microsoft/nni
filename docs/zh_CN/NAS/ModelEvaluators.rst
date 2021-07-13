模型 Evaluator
================

模型评估器（model evaluator）用于训练和验证每个生成的模型。

模型评估器的用法
------------------------------------

在 multi-NAS 中，采样模型应该能够在远程机器或训练平台（例如 AzureML、OpenPAI）上执行。 因此，模型及其模型评估器都应该正确序列化。 为了使 NNI 正确地序列化模型评估器，用户应该在他们的一些函数和对象上应用 ``serialize`` 。

.. _serializer:

`serialize <./ApiReference.rst#id7>`__ 允许在另一个进程或机器中重新实例化模型评估器。 它是通过记录用户实例化的评估器的初始化参数来实现的。

Retiarii 提供的评估器相关 API 已经支持序列化，例如 ``pl.Classification``, ``pl.DataLoader``，无需对其应用 ``serialize``。 在以下情况下，用户应该手动使用 ``serialize`` API。

如果评估器 API 的初始化参数（例如 ``pl.Classification``、``pl.DataLoader``）不是原始类型（例如 ``int``, ``string``），它们应该是与 ``serialize`` 一起应用。 如果这些参数的初始化参数不是原始类型，``serialize`` 也应该被应用。 总而言之，如果有必要，``serialize`` 应该被递归应用。

以下是一个示例，``transforms.Compose``, ``transforms.Normalize`` 和 ``MNIST`` 应该通过 ``serialize`` 手动序列化。 ``serialize`` 以一个类 ``cls`` 作为它的第一个参数，它后面的参数是初始化这个类的参数。 ``pl.Classification`` 没有应用 ``serialize`` 因为它已经被 NNI 提供的 API 序列化。

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

支持的模型评估器
----------------------------------------

NNI 提供了一些常用的模型评估器以方便用户使用。 如果这些模型评估器不满足用户的要求，您可以按照 `教程 <./WriteTrainer.rst>`__ 自定义新的模型评估器。

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Classification
    :noindex:

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Regression
    :noindex:
