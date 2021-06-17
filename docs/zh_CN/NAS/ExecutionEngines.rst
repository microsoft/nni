Execution Engines
=================

Execution engine is for running Retiarii Experiment. NNI supports three execution engines, users can choose a speicific engine according to the type of their model mutation definition and their requirements for cross-model optimizations. 

* **Pure-python execution engine** is the default engine, it supports the model space expressed by `inline mutation API <./MutationPrimitives.rst>`__. 

* **Graph-based execution engine** supports the use of `inline mutation APIs <./MutationPrimitives.rst>`__ and model spaces represented by `mutators <./Mutators.rst>`__. It requires the user's model to be parsed by `TorchScript <https://pytorch.org/docs/stable/jit.html>`__.

* **CGO execution engine** has the same requirements and capabilities as the **Graph-based execution engine**. But further enables cross-model optimizations, which makes model space exploration faster.

Pure-python Execution Engine
----------------------------

Pure-python Execution Engine is the default engine, we recommend users to keep using this execution engine, if they are new to NNI NAS. Pure-python execution engine plays magic within the scope of inline mutation APIs, while does not touch the rest of user model. Thus, it has minimal requirement on user model. 

One steps are needed to use this engine now.

1. Add ``@nni.retiarii.model_wrapper`` decorator outside the whole PyTorch model.

.. note:: You should always use ``super().__init__()`` instead of ``super(MyNetwork, self).__init__()`` in the PyTorch model, because the latter one has issues with model wrapper.

Graph-based Execution Engine
----------------------------

For graph-based execution engine, it converts user-defined model to a graph representation (called graph IR) using `TorchScript <https://pytorch.org/docs/stable/jit.html>`__, each instantiated module in the model is converted to a subgraph. Then mutations are applied to the graph to generate new graphs. Each new graph is then converted back to PyTorch code and executed on the user specified training service.

Users may find ``@basic_unit`` helpful in some cases. ``@basic_unit`` here means the module will not be converted to a subgraph, instead, it is converted to a single graph node as a basic unit.

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

Three steps are need to use graph-based execution engine.

1. Remove ``@nni.retiarii.model_wrapper`` if there is any in your model.
2. Add ``config.execution_engine = 'base'`` to ``RetiariiExeConfig``. The default value of ``execution_engine`` is 'py', which means pure-python execution engine.
3. Add ``@basic_unit`` when necessary following the above guidelines.

For exporting top models, graph-based execution engine supports exporting source code for top models by running ``exp.export_top_models(formatter='code')``.

CGO Execution Engine
--------------------

CGO execution engine does cross-model optimizations based on the graph-based execution engine. This execution engine will be `released in v2.4 <https://github.com/microsoft/nni/issues/3813>`__.
