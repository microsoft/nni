NNI 客户端
==========

NNI 客户端是 ``nnictl`` 的python API，提供了对常用命令的实现。 相比于命令行，用户可以通过此 API 来在 python 代码中直接操控实验，收集实验结果并基于实验结果进行更加高级的分析。 示例如下：

.. code-block:: bash

   from nni.experiment import Experiment

   # 创建一个实验实例
   exp = Experiment() 

   # 开始一个实验，将实例与实验连接
   # 你也可以使用 `resume_experiment`, `view_experiment` 或者 `connect_experiment`
   # 在一个实例中只能调用其中一个
   exp.start_experiment('nni/examples/trials/mnist-pytorch/config.yml', port=9090)

   # 更新实验的并发数量
   exp.update_concurrency(3)

   # 获取与实验相关的信息
   print(exp.get_experiment_status())
   print(exp.get_job_statistics())
   print(exp.list_trial_jobs())

   # 停止一个实验，将实例与实验断开
   exp.stop_experiment()

参考
----------

..  autoclass:: nni.experiment.Experiment
    :members:
..  autoclass:: nni.experiment.TrialJob
    :members:
..  autoclass:: nni.experiment.TrialHyperParameters
    :members:
..  autoclass:: nni.experiment.TrialMetricData
    :members:
..  autoclass:: nni.experiment.TrialResult
    :members:
