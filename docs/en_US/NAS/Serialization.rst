Serialization
=============

In multi-trial NAS, a sampled model should be able to be executed on a remote machine or a training platform (e.g., AzureML, OpenPAI). "Serialization" enables re-instantiation of model evaluator in another process or machine, such that, both the model and its model evaluator should be correctly serialized. To make NNI correctly serialize model evaluator, users should apply ``nni.trace`` on some of their functions and objects. API references can be found in :func:`nni.trace`.

Serialization is implemented as a combination of `json-tricks <https://json-tricks.readthedocs.io/en/latest/>`_ and `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_. Essentially, it is json-tricks, that is a enhanced version of Python JSON, enabling handling of serialization of numpy arrays, date/times, decimal, fraction and etc. The difference lies in the handling of class instances. Json-tricks deals with class instances with ``__dict__`` and ``__class__``, which in most of our cases are not reliable (e.g., datasets, dataloaders). Rather, our serialization deals with class instances with two methods:

1. If the class / factory that creates the object is decorated with ``nni.trace``, we can serialize the class / factory function, along with the parameters, such that the instance can be re-instantiated.
2. Otherwise, cloudpickle is used to serialize the object into a binary.

The recommendation is, unless you are absolutely certain that there is no problem and extra burden to serialize the object into binary, always add ``nni.trace``. In most cases, it will be more clean and neat, and enables possibilities such as mutation of parameters (will be supported in future).

.. warning::

    **What will happen if I forget to "trace" my objects?**

    It is likely that the program can still run. NNI will try to serialize the untraced object into a binary. If might fail in complicated cases (e.g., circular dependency). Even if it succeeds, the result might be a substantially large object. For example, if you forgot to add ``nni.trace`` on ``MNIST``, the MNIST dataset object wil be serialized into binary, which will be dozens of megabytes because the object has the whole 60k images stored inside. You might see warnings and even errors when running experiments. To avoid such issues, the easiest way is to always remember to add ``nni.trace`` to non-primitive objects.

To trace a function or class, users can use decorator like,

.. code-block:: python

    @nni.trace
    class MyClass:
        ...

Inline trace that traces instantly on the object instantiation or function invoke is also acceptable: ``nni.trace(MyClass)(parameters)``.

Assuming a class ``cls`` is already traced, when it is serialized, its class type along with initialization parameters will be dumped. As the parameters are possibly class instances (if not primitive types like ``int`` and ``str``), their serialization will be a similar problem. We recommend decorate them with ``nni.trace`` as well. In other words, ``nni.trace`` should be applied recursively if necessary.

Below is an example, ``transforms.Compose``, ``transforms.Normalize``, and ``MNIST`` are serialized manually using ``nni.trace``. ``nni.trace`` takes a class / function as its argument, and returns a wrapped class and function that has the same behavior with the original class / function. The usage of the wrapped class / function is also identical to the original one, except that the arguments are recorded. No need to apply ``nni.trace`` to ``pl.Classification`` and ``pl.DataLoader`` because they are already traced.

.. code-block:: python

  import nni
  import nni.retiarii.evaluator.pytorch.lightning as pl
  from torchvision import transforms

  def create_mnist_dataset(root, transform):
    return MNIST(root='data/mnist', train=False, download=True, transform=transform)

  transform = nni.trace(transforms.Compose)([nni.trace(transforms.ToTensor)(), nni.trace(transforms.Normalize)((0.1307,), (0.3081,))])

  # If you write like following, the whole transform will be serialized into a pickle.
  # This actually works fine, but we do NOT recommend such practice.
  # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

  train_dataset = nni.trace(MNIST)(root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = nni.trace(create_mnist_dataset)('data/mnist', transform=transform)  # factory is also acceptable
  evaluator = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=10)

.. note::

    **What's the relationship between model_wrapper, basic_unit and nni.trace?**

    They are fundamentally different. ``model_wrapper`` is used to wrap a base model (search space), ``basic_unit`` to annotate a module as primitive. ``nni.trace`` is to enable serialization of general objects. Though they share similar underlying implementations, but do keep in mind that you will experience errors if you mix them up.

    .. seealso:: Please refer to API reference of :meth:`nni.retiarii.model_wrapper`, :meth:`nni.retiarii.basic_unit`, and :meth:`nni.trace`.
