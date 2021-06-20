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

对于基于图形的执行引擎，它使用 `TorchScript <https://pytorch.org/docs/stable/jit.html>`__ 将用户定义的模型转换为图形表示（称为图形 IR），其中的每个实例化模块模型转换为子图。 然后将突变应用到图上，生成新的图。 每个新图被转换回 PyTorch 代码并在用户指定的训练服务上执行。

在某些情况下，用户可能会发现 ``@basic_unit`` 非常有帮助。 ``@basic_unit`` 这里意味着模块将不会被转换为子图，而是将其转换为单个图形节点作为基本单元。

``@basic_unit`` 通常在以下情况下使用：

* 当用户想要调整模块的初始化参数时使用 ``valueChoice``，然后用 ``@ basic_unit`` 装饰模块。 例如，在 ``self.conv = MyConv(kernel_size=nn.ValueChoice([1, 3, 5]))`` 中，``MyConv`` 应该被修饰。

* 当模块无法被成功解析为子图，用 ``@basic_unit`` 装饰模块。 解析失败可能是由于复杂的控制流造成的。 目前 Retiarii 不支持 adhoc 循环，如果在一个模块的前向有 adhoc 循环，这个类应该被装饰成可序列化的模块。 例如，下面的 ``MyModule`` 应该被装饰起来。

  .. code-block:: python

    @basic_unit
    class MyModule(nn.Module):
      def __init__(self):
        ...
      def forward(self, x):
        for i in range(10): # <- adhoc loop
          ...

* 一些内联突变 API 要求其处理的模块用 ``@basic_unit`` 来装饰。 例如，提供给 ``LayerChoice`` 的用户自定义的模块作为候选操作应该被装饰。

使用基于图的执行引擎需要三个步骤：

1. 如果你的模型中有 ``@nni.retiarii.model_wrapper``，请移除。
2. 将 ``config.execution_engine = 'base'`` 添加到 ``RetiariiExeConfig``。 ``execution_engine`` 的默认值是 'py'，即纯 Python 执行引擎。
3. 必要时按照上述准则添加 ``@basic_unit``。

对于导出最佳模型，基于图的执行引擎支持通过运行 ``exp.export_top_models(formatter='code')`` 来导出最佳模型的源代码。

CGO 执行引擎
--------------------

CGO 执行引擎在基于图的执行引擎基础上进行跨模型的优化。 这个执行引擎将在 `v2.4 发布 <https://github.com/microsoft/nni/issues/3813>`__。
