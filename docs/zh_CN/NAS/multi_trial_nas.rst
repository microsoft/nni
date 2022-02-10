.. 51734c9945d4eca0f9b5633929d8fadf

Multi-trial NAS
===============

在 multi-trial NAS 中，用户需要模型评估器来评估每个采样模型的性能，并且需要一个探索策略来从定义的模型空间中采样模型。 在这里，用户可以使用 NNI 提供的模型评估器或编写自己的模型评估器。 他们可以简单地选择一种探索策略。 高级用户还可以自定义新的探索策略。 关于如何运行 multi-trial NAS 实验的简单例子，请参考 `快速入门 <./QuickStart.rst>`__。

..  toctree::
    :maxdepth: 1

    模型评估器 <ModelEvaluators>
    探索策略 <ExplorationStrategies>
    执行引擎 <ExecutionEngines>
    序列化 <Serialization>
