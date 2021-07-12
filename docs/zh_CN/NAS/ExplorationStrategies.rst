Multi-trial NAS 的探索策略
=====================================================

探索策略的用法
-----------------------------

要使用探索策略，用户只需将探索策略实例化，并将实例化的对象传递给 ``RetiariiExperiment``。 示例如下：

.. code-block:: python

  import nni.retiarii.strategy as strategy

  exploration_strategy = strategy.Random(dedup=True)  # dedup=False 如果不希望有重复数据删除

支持的探索策略
--------------------------------

NNI 提供了以下 multi-trial NAS 的探索策略。 用户还可以 `自定义新的探索策略 <./WriteStrategy.rst>`__。

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 名字
     - 算法简介
   * - `随机策略 <./ApiReference.rst#nni.retiarii.strategy.Random>`__
     - 从搜索空间中随机选择模型 (``nni.retiarii.strategy.Random``)
   * - `网格搜索 <./ApiReference.rst#nni.retiarii.strategy.GridSearch>`__
     - 使用网格搜索算法从用户定义的模型空间中采样新模型。 (``nni.retiarii.strategy.GridSearch``)
   * - `正则进化 <./ApiReference.rst#nni.retiarii.strategy.RegularizedEvolution>`__
     - 使用 `正则进化算法 <https://arxiv.org/abs/1802.01548>`__ 从生成的模型中生成新模型 (``nni.retiarii.strategy.RegularizedEvolution``)
   * - `TPE 策略 <./ApiReference.rst#nni.retiarii.strategy.TPEStrategy>`__
     - 使用 `TPE 算法 <https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>`__ 从用户定义的模型空间中生成新模型 (``nni.retiarii.strategy.TPEStrategy``)
   * - `RL 策略 <./ApiReference.rst#nni.retiarii.strategy.PolicyBasedRL>`__
     - 使用 `PPO 算法 <https://arxiv.org/abs/1707.06347>`__ 从用户定义的模型空间中生成新模型 (``nni.retiarii.strategy.PolicyBasedRL``)