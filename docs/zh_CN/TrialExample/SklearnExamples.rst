NNI 中使用 scikit-learn
======================================

`Scikit-learn <https://github.com/scikit-learn/scikit-learn>`__ (sklearn) 是流行的数据挖掘和分析工具。 它支持多种机器学习模型，如线性回归，逻辑回归，决策树，支持向量机等。 如何更高效的使用 scikit-learn，是一个很有价值的话题。

NNI 支持多种调优算法来为 scikit-learn 搜索最好的模型和超参，并支持本机、远程服务器和云服务等多种环境。

1. 如何运行此示例
-------------------------

安装 NNI 包，并使用命令行工具 ``nnictl`` 来启动 Experiment。 有关安装和环境准备的内容，参考 `这里 <../Tutorial/QuickStart.rst>`__。

安装完 NNI 后，进入相应的目录，输入下列命令即可启动 Experiment：

.. code-block:: bash

   nnictl create --config ./config.yml

2. 示例概述
-----------------------------

2.1 分类
^^^^^^^^^^^^^^^^^^

示例使用了数字数据集，它是由 1797 个 8x8 的图片组成，每个图片都是一个手写数字，目标是将图片分为 10 类。

在这个示例中，使用 SVC 作为模型，并为此模型选择一些参数，包括 ``"C", "kernel", "degree", "gamma" 和 "coef0"``。 关于这些参数的更多信息，可参考 `这里 <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`__。

2.2 回归
^^^^^^^^^^^^^^

此示例使用了波士顿房价数据，数据集由波士顿各地区房价所组成，还包括了房屋的周边信息，例如：犯罪率 (CRIM)，非零售业务的面积 (INDUS)，房主年龄 (AGE) 等等。这些信息可用来预测波士顿的房价。

本例中，尝试了不同的回归模型，包括 ``"LinearRegression", "SVR", "KNeighborsRegressor", "DecisionTreeRegressor"</code> 和一些参数，如 <code>"svr_kernel", "knr_weights"``。 关于这些模型算法和参数的更多信息，可参考 `这里 <https://scikit-learn.org/stable/supervised_learning.html#supervised-learning>`__ 。

3. 如何在 NNI 中使用 scikit-learn
-------------------------------------------

只需要如下几步，即可在 scikit-learn 代码中使用 NNI。


* 
  **第一步**

  准备 search_space.json 文件来存储选择的搜索空间。
  例如，如果要在不同的模型中选择：

  .. code-block:: json

     {
       "model_name":{"_type":"choice","_value":["LinearRegression", "SVR", "KNeighborsRegressor", "DecisionTreeRegressor"]}
     }

  如果要选择不同的模型和参数，可以将它们放到同一个 search_space.json 文件中。

  .. code-block:: json

     {
       "model_name":{"_type":"choice","_value":["LinearRegression", "SVR", "KNeighborsRegressor", "DecisionTreeRegressor"]},
       "svr_kernel": {"_type":"choice","_value":["linear", "poly", "rbf"]},
       "knr_weights": {"_type":"choice","_value":["uniform", "distance"]}
     }

  在 Python 代码中，可以将这些值作为一个 dict，读取到 Python 代码中。

* 
  **第二步**

  在代码最前面，加上 ``import nni`` 来导入 NNI 包。

  首先，要使用 ``nni.get_next_parameter()`` 函数从 NNI 中获取参数。 然后在代码中使用这些参数。
  例如，如果定义了如下的 search_space.json：

  .. code-block:: json

     {
       "C": {"_type":"uniform","_value":[0.1, 1]},
       "kernel": {"_type":"choice","_value":["linear", "rbf", "poly", "sigmoid"]},
       "degree": {"_type":"choice","_value":[1, 2, 3, 4]},
       "gamma": {"_type":"uniform","_value":[0.01, 0.1]},
       "coef0": {"_type":"uniform","_value":[0.01, 0.1]}
     }

  就会获得像下面一样的 dict：

  .. code-block:: python

     params = {
           'C': 1.0,
           'kernel': 'linear',
           'degree': 3,
           'gamma': 0.01,
           'coef0': 0.01
     }

  就可以使用这些变量来编写 scikit-learn 的代码。

* 
  **第三步**

  完成训练后，可以得到模型分数，如：精度，召回率，均方差等等。 NNI 需要将分数传入 Tuner 算法，并生成下一组参数，将结果回传给 NNI，并开始下一个 Trial 任务。

  在运行完 scikit-learn 代码后，只需要使用 ``nni.report_final_result(score)`` 来与 NNI 通信即可。 或者在每一步中都有多个分值，可使用 ``nni.report_intemediate_result(score)`` 来将它们回传给 NNI。 注意， 可以不返回中间分数，但必须返回最终的分数。
