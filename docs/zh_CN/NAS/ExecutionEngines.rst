执行引擎
=================

执行引擎（Execution engine）用于运行 Retiarii 实验。 NNI 支持三种执行引擎，用户可以根据自己的模型 mutation 定义类型和跨模型优化的需求选择特定的引擎。 

* **纯 Python 执行引擎（Pure-python execution engine）** 是默认引擎，它支持 `内联 mutation API <./MutationPrimitives.rst>`__ 表示的模型空间。 

* **基于图的执行引擎（Graph-based execution engine）** 支持使用 `内联 mutation APIs <./MutationPrimitives.rst>`__ 和由 `mutators <./Mutators.rst>`__ 表示的模型空间。 它要求用户的模型由 `TorchScript <https://pytorch.org/docs/stable/jit.html>`__ 解析。

* **CGO执行引擎（CGO execution engine）** 具有与 **基于图形的执行引擎** 相同的要求和能力。 但未来将支持跨模型的优化，这使得模型空间的探索变得更快。

纯 Python 执行引擎
----------------------------

纯 Python 执行引擎是默认引擎，如果用户是 NNI NAS 的新手，我们建议使用这个执行引擎。 纯 Python 执行引擎在内联突变 API 的范围内发挥了神奇作用，而不会触及用户模型的其余部分。 因此，它对用户模型的要求最低。 

现在需要一个步骤来使用这个引擎。

1. 在整个 PyTorch 模型之外添加 ``@nni.retiarii.model_wrapper`` 装饰器。

.. note:: 在 PyTorch 模型中，您应该始终使用 ``super().__init__()`` 而不是 ``super(MyNetwork, self).__init__()`` ，因为后者在模型装饰器上存在问题。

基于图的执行引擎
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
