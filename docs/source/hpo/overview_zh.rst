.. c74f6d072f5f8fa93eadd214bba992b4

超参调优
========

自动超参调优（hyperparameter optimization, HPO）是 NNI 的主要功能之一。

超参调优简介
------------

在机器学习中，用来控制学习过程的参数被称为“超参数”或“超参”，而为一种机器学习算法选择最优超参组合的问题被称为“超参调优”。

以下代码片段演示了一次朴素的超参调优：

.. code-block:: python

    best_hyperparameters = None
    best_accuracy = 0

    for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
        for momentum in [i / 10 for i in range(10)]:
            for activation_type in ['relu', 'tanh', 'sigmoid']:
                model = build_model(activation_type)
                train_model(model, learning_rate, momentum)
                accuracy = evaluate_model(model)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyperparameters = (learning_rate, momentum, activation_type)

    print('最优超参：', best_hyperparameters)

可以看到，这段超参调优代码总计训练4×10×3=120个模型，要消耗大量的计算资源，因此您可能会有以下需求：

1. :ref:`通过较少的尝试次数找到最优超参组合 <zh-hpo-overview-tuners>`
2. :ref:`利用分布式平台进行训练 <zh-hpo-overview-platforms>`
3. :ref:`使用网页控制台来监控调参过程 <zh-hpo-overview-portal>`

NNI 可以满足您的这些需求。

NNI 超参调优的主要功能
----------------------

.. _zh-hpo-overview-tuners:

调优算法
^^^^^^^^

NNI 通过调优算法来更快地找到最优超参组合，这些算法被称为“tuner”（调参器）。

调优算法会决定需要运行、评估哪些超参组合，以及应该以何种顺序评估超参组合。
高效的算法可以通过已评估超参组合的结果去预测最优超参的取值，从而减少找到最优超参所需的评估次数。

开头的示例以固定顺序评估所有可能的超参组合，无视了超参的评估结果，这种朴素方法被称为“grid search”（网格搜索）。

NNI 内建了很多流行的调优算法，包括朴素算法如随机搜索、网格搜索，贝叶斯优化类算法如 TPE、SMAC，强化学习算法如 PPO 等等。

完整内容： :doc:`tuners`

.. _zh-hpo-overview-platforms:

训练平台
^^^^^^^^

如果您不准备使用分布式训练平台，您可以像使用普通 Python 函数库一样，在自己的电脑上直接运行 NNI 超参调优。

如果想利用更多计算资源加速调优过程，您也可以使用 NNI 内建的训练平台集成，从简单的 SSH 服务器到可扩容的 Kubernetes 集群 NNI 都提供支持。

完整内容： :doc:`/experiment/training_service/overview`

.. _zh-hpo-overview-portal:

网页控制台
^^^^^^^^^^

您可以使用 NNI 的网页控制台来监控超参调优实验，它支持实时显示实验进度、对超参性能进行可视化、人工修改超参数值、同时管理多个实验等诸多功能。

完整内容： :doc:`/experiment/web_portal/web_portal`

.. image:: ../../static/img/webui.gif
    :width: 100%

教程
----

我们提供了以下教程帮助您上手 NNI 超参调优，您可以选择最熟悉的机器学习框架：

* :doc:`使用PyTorch的超参调优教程 </tutorials/hpo_quickstart_pytorch/main>`
* :doc:`使用TensorFlow的超参调优教程（英文） </tutorials/hpo_quickstart_tensorflow/main>`

更多功能
--------

在掌握了 NNI 超参调优的基础用法之后，您可以尝试以下更多功能：

* :doc:`Use command line tool to create and manage experiments (nnictl) </reference/nnictl>`

  * :doc:`nnictl example </tutorials/hpo_nnictl/nnictl>`

* :doc:`Early stop non-optimal models (assessor) <assessors>`
* :doc:`TensorBoard integration </experiment/web_portal/tensorboard>`
* :doc:`Implement your own algorithm <custom_algorithm>`
* :doc:`Benchmark tuners <hpo_benchmark>`
