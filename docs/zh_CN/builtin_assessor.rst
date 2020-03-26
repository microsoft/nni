内置 Assessor
=================

为了节省计算资源，NNI 支持提前终止策略，并且通过叫做 **Assessor** 的接口来执行此操作。

Assessor 从 Trial 中接收中间结果，并通过指定的算法决定此 Trial 是否应该终止。 一旦 Trial 满足了提前终止策略（这表示 Assessor 认为最终结果不会太好），Assessor 会终止此 Trial，并将其状态标志为 `EARLY_STOPPED`。

Here is an experimental result of MNIST after using the 'Curvefitting' Assessor in 'maximize' mode. You can see that Assessor successfully **early stopped** many trials with bad hyperparameters in advance. If you use Assessor, you may get better hyperparameters using the same computing resources.

*实现代码：[config_assessor.yml](https://github.com/Microsoft/nni/blob/master/examples/trials/mnist-tfv1/config_assessor.yml) *

..  image:: ../img/Assessor.png

..  toctree::
    :maxdepth: 1

    概述<./Assessor/BuiltinAssessor>
    Medianstop<./Assessor/MedianstopAssessor>
    Curvefitting（曲线拟合）<./Assessor/CurvefittingAssessor>