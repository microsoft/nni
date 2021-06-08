``@basic_unit`` and ``serializer``
==================================

.. _serializer:

``@basic_unit`` and ``serialize`` can be viewed as some kind of serializer. They are designed for making the whole model (including training) serializable to be executed on another process or machine.

**@basic_unit** annotates that a module is a basic unit, i.e, no need to understand the details of this module. The effect is that it prevents Retiarii to parse this module. To understand this, we first briefly explain how Retiarii works: it converts user-defined model to a graph representation (called graph IR) using `TorchScript <https://pytorch.org/docs/stable/jit.html>`__, each instantiated module in the model is converted to a subgraph. Then mutations are applied to the graph to generate new graphs. Each new graph is then converted back to PyTorch code and executed. ``@basic_unit`` here means the module will not be converted to a subgraph, instead, it is converted to a single graph node as a basic unit. That is, the module will not be unfolded anymore. When the module is not unfolded, mutations on initialization parameters of this module becomes easier.

``@basic_unit`` is usually used in the following cases:

* When users want to tune initialization parameters of a module using ``ValueChoice``, then decorate the module with ``@basic_unit``. For example, ``self.conv = MyConv(kernel_size=nn.ValueChoice([1, 3, 5]))``, here ``MyConv`` should be decorated.

* When a module cannot be successfully parsed to a subgraph, decorate the module with ``@basic_unit``. The parse failure could be due to complex control flow. Currently Retiarii does not support adhoc loop, if there is adhoc loop in a module's forward, this class should be decorated as serializable module. For example, the following ``MyModule`` should be decorated.

  .. code-block:: python

    @basic_unit
    class MyModule(nn.Module):
      def __init__(self):
        ...
      def forward(self, x):
        for i in range(10): # <- adhoc loop
          ...

* Some inline mutation APIs require their handled module to be decorated with ``@basic_unit``. For example, user-defined module that is provided to ``LayerChoice`` as a candidate op should be decorated.

**serialize** is mainly used for serializing model training logic. It enables re-instantiation of model evaluator in another process or machine. Re-instantiation is necessary because most of time model and evaluator should be sent to training services. ``serialize`` is implemented by recording the initialization parameters of user instantiated evaluator.

The evaluator related APIs provided by Retiarii have already supported serialization, for example ``pl.Classification``, ``pl.DataLoader``, no need to apply ``serialize`` on them. In the following case users should use ``serialize`` API manually.

If the initialization parameters of the evaluator APIs (e.g., ``pl.Classification``, ``pl.DataLoader``) are not primitive types (e.g., ``int``, ``string``), they should be applied with  ``serialize``. If those parameters' initialization parameters are not primitive types, ``serialize`` should also be applied. In a word, ``serialize`` should be applied recursively if necessary.
