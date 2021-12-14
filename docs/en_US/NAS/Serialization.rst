Serialization
=============

`serialize <./ApiReference.rst#utilities>`__ enables re-instantiation of model evaluator in another process or machine. It is implemented by recording the initialization parameters of user instantiated evaluator.

The evaluator related APIs provided by Retiarii have already supported serialization, for example ``pl.Classification``, ``pl.DataLoader``, no need to apply ``serialize`` on them. In the following case users should use ``serialize`` API manually.

If the initialization parameters of the evaluator APIs (e.g., ``pl.Classification``, ``pl.DataLoader``) are not primitive types (e.g., ``int``, ``string``), they should be applied with  ``serialize``. If those parameters' initialization parameters are not primitive types, ``serialize`` should also be applied. In a word, ``serialize`` should be applied recursively if necessary.

Below is an example, ``transforms.Compose``, ``transforms.Normalize``, and ``MNIST`` are serialized manually using ``serialize``. ``serialize`` takes a class ``cls`` as its first argument, its following arguments are the arguments for initializing this class. ``pl.Classification`` is not applied ``serialize`` because it is already serializable as an API provided by NNI.
