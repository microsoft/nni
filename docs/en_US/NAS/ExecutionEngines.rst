Execution Engines
=================

Execution engine is for running NAS experiment. NNI supports several execution engines, each of them has their own characteristics. **Pure-python execution engine** does not have special requirement on user model, it only supports model space expressed with inline mutation APIs. **Graph-based execution engine** requires that user's model should be able to be parsed by TorchScript, it supports model space expressed with both inline mutation APIs and mutators. **CGO execution engine** has the same requirement and ability to graph-based execution engine, it further enables cross-model optimizations, which makes model space exploration faster.

Pure-python Execution Engine
----------------------------

If you are experiencing issues with TorchScript, or the generated model code by Retiarii, there is another execution engine called Pure-python execution engine which doesn't need the code-graph conversion. This should generally not affect models and strategies in most cases, but customized mutation might not be supported.

This will come as the default execution engine in future version of Retiarii.

Three steps are needed to enable this engine now.

1. Add ``@nni.retiarii.model_wrapper`` decorator outside the whole PyTorch model.
2. Add ``config.execution_engine = 'py'`` to ``RetiariiExeConfig``.
3. If you need to export top models, formatter needs to be set to ``dict``. Exporting ``code`` won't work with this engine.

.. note:: You should always use ``super().__init__()` instead of ``super(MyNetwork, self).__init__()`` in the PyTorch model, because the latter one has issues with model wrapper.

Graph-based Execution Engine
----------------------------

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


CGO Execution Engine
--------------------

CGO execution engine does cross-model optimizations based on the graph-based execution engine. This execution engine will be release in v2.4.