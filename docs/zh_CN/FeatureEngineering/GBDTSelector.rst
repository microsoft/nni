GBDTSelector
------------

GBDTSelector 基于 `LightGBM <https://github.com/microsoft/LightGBM>`__，这是一个基于树学习算法的梯度提升框架。

当将数据传递到 GBDT 模型时，该模型将构建提升树。 特征的重要性来自于构造时的分数，其表达了每个特征在模型构造提升决策树时有多有用。

可使用此方法作为 Feature Selector 中较强的基准，特别是在使用 GBDT 模型进行分类或回归时。

当前，支持的 ``importance_type`` 有 ``split`` 和 ``gain``。 未来会支持定制 ``importance_type``，也就是说用户可以定义如何计算 ``特征分数``。

用法
^^^^^

首先，安装依赖项：

.. code-block:: bash

   pip install lightgbm

然后

.. code-block:: python

   from nni.feature_engineering.gbdt_selector import GBDTSelector

   # 下载数据
   ...
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

   # 初始化 selector
   fgs = GBDTSelector()
   # 拟合数据
   fgs.fit(X_train, y_train, ...)
   # 获取重要的特征
   # 此处会返回重要特征的索引。
   print(fgs.get_selected_features(10))

   ...

也可在 ``/examples/feature_engineering/gbdt_selector/`` 目录找到示例。

``fit`` 函数参数要求


* 
  **X** (数组，必需) - 训练的输入样本，shape = [n_samples, n_features]

* 
  **y** (数组，必需) - 目标值 (分类中为标签，回归中为实数)，shape = [n_samples].

* 
  **lgb_params** (dict, 必需) - lightgbm 模型参数。 详情参考 `这里 <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`__

* 
  **eval_ratio** (float, 必需) - 数据大小的比例 用于从 self.X 中拆分出评估和训练数据。

* 
  **early_stopping_rounds** (int, 必需) - lightgbm 中的提前终止设置。 详情参考 `这里 <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`__。

* 
  **importance_type** (str, 必需) - 可为 'split' 或 'gain'。 'split' 表示 '结果包含特征在模型中使用的次数' 而 'gain' 表示 '结果包含此特征拆分出的总收益'。 详情参考 `这里 <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance>`__。

* 
  **num_boost_round** (int, 必需) - 提升的轮数。 详情参考 `这里 <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm.train>`__.

**get_selected_features 函数参数的要求**


**topk** (int, 必需) - 想要选择的 k 个最好的特征。
