.. role:: raw-html(raw)
   :format: html


内置 Assessor
==================

NNI 提供了先进的评估算法，使用上也很简单。 下面是内置 Assessor 的介绍。

注意：点击 **Assessor 的名称** 可了解每个 Assessor 的安装需求，建议的场景以及示例。 在每个 Assessor 建议场景最后，还有算法的详细说明。

当前支持以下 Assessor：

.. list-table::
   :header-rows: 1
   :widths: auto

   * -  Assessor 
     - 算法简介
   * - `Medianstop <#MedianStop>`__
     - Medianstop 是一个简单的提前终止算法。 如果尝试 X 的在步骤 S 的最好目标值比所有已完成尝试的步骤 S 的中位数值明显低，就会停止运行尝试 X。 `参考论文 <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf>`__
   * - `Curvefitting <#Curvefitting>`__
     - Curve Fitting Assessor 是一个 LPA (learning, predicting, assessing，即学习、预测、评估) 的算法。 如果预测的 Trial X 在 step S 比性能最好的 Trial 要差，就会提前终止它。 此算法中采用了 12 种曲线来拟合精度曲线。 `参考论文 <http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf>`__


用法
--------------------------

要使用 NNI 内置的 Assessor，需要在 ``config.yml`` 文件中添加 **builtinAssessorName** 和 **classArgs**。 这一节会介绍推荐的场景、参数等详细用法以及示例。

注意：参考示例中的格式来创建新的 ``config.yml`` 文件。

:raw-html:`<a name="MedianStop"></a>`

Median Stop Assessor
^^^^^^^^^^^^^^^^^^^^

..

   名称：**Medianstop**


**建议场景**

适用于各种性能曲线，可用到各种场景中来加速优化过程。 `详细说明 <./MedianstopAssessor.rst>`__

**classArgs 要求：**


* **optimize_mode** (*maximize 或 minimize，可选默认值是maximize*)。如果为 'maximize'，Assessor 会在结果小于期望值时**中止** Trial。 如果为 'minimize'，Assessor 会在结果大于期望值时**终止** Trial。
* **start_step** (*int，可选，默认值为 0*)。只有收到 start_step 个中间结果后，才开始判断是否一个 Trial 应该被终止。

**使用示例：**

.. code-block:: yaml

   # config.yml
   assessor:
       builtinAssessorName: Medianstop
       classArgs:
         optimize_mode: maximize
         start_step: 5

:raw-html:`<br>`

:raw-html:`<a name="Curvefitting"></a>`

Curve Fitting Assessor
^^^^^^^^^^^^^^^^^^^^^^

..

   名称：**Curvefitting**


**建议场景**

适用于各种性能曲线，可用到各种场景中来加速优化过程。 更好的是，它能够处理并评估性能类似的曲线。 `详细说明 <./CurvefittingAssessor.rst>`__

**注意**，根据原始论文，仅支持递增函数。 因此，此 Assessor 仅可用于最大化优化指标的场景。 例如，它可用于准确度，但不能用于损失值。

**classArgs 要求：**


* **epoch_num** (*int，必需*)，epoch 的总数。 需要此数据来决定需要预测点的总数。
* **start_step** (*int，可选，默认值为 6*)。只有收到 start_step 个中间结果后，才开始判断是否一个 Trial 应该被终止。
* **threshold** (*float，可选，默认值为 0.95*)，用来确定提前终止较差结果的阈值。 例如，如果 threshold = 0.95，最好的历史结果是 0.9，那么会在 Trial 的预测值低于 0.95 * 0.9 = 0.855 时停止。
* **gap** (*int，可选，默认值为 1*)，Assessor 两次评估之间的间隔次数。 例如：如果 gap = 2, start_step = 6，就会评估第 6, 8, 10, 12... 个中间结果。

**使用示例：**

.. code-block:: yaml

   # config.yml
   assessor:
       builtinAssessorName: Curvefitting
       classArgs:
         epoch_num: 20
         start_step: 6
         threshold: 0.95
         gap: 1
