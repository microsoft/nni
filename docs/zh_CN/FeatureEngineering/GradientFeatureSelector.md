## GradientFeatureSelector

GradinetFeatureSelector 的算法来源于 ["Feature Gradients: Scalable Feature Selection via Discrete Relaxation"](https://arxiv.org/pdf/1908.10382.pdf)。

GradientFeatureSelector，基于梯度搜索算法的特征选择。

1) 该方法扩展了一个近期的结果，即在亚线性数据中通过展示计算能迭代的学习（即，在迷你批处理中），在**线性的时间空间中**的特征数量 D 及样本大小 N。

2) 这与在搜索领域的离散到连续的放松一起，可以在非常**大的数据集**上进行**高效、基于梯度**的搜索算法。

3) 最重要的是，此算法能在特征和目标间为 N > D 和 N < D 都找到**高阶相关性**，这与只考虑一种情况和交互式的方法所不同。


### 用法

```python
from nni.feature_engineering.gradient_selector import FeatureGradientSelector

# 读取数据
...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 初始化 Selector
fgs = FeatureGradientSelector()
# 拟合数据
fgs.fit(X_train, y_train)
# 获取重要的特征
# 此处会返回重要特征的索引。
print(fgs.get_selected_features())

...
```

也可在 `/examples/feature_engineering/gradient_feature_selector/` 目录找到示例。

**FeatureGradientSelector 构造函数的参数**

* **order** (int, 可选, 默认为 4) - 要包含的交互顺序。 较高的顺序可能会更准确，但会增加运行时间。 12 是允许的顺序的最大值。

* **penatly** (int, 可选, 默认为 1) - 乘以正则项的常数。

* **n_features** (int, 可选, 默认为 None) - 如果为 None，会自动根据搜索来选择特征的数量。 否则，表示要选择的最好特征的数量。

* **max_features** (int, 可选, 默认为 None) - 如果不为 None，会使用 'elbow method' 来确定以 max_features 为上限的特征数量。

* **learning_rate** (float, 可选, 默认为 1e-1) - 学习率

* **init** (*zero, on, off, onhigh, offhigh, 或 sklearn, 可选, 默认为zero*) - 如何初始化向量分数。 默认值为 'zero'。

* **n_epochs** (int, 可选, 默认为 1) - 要运行的 Epoch 数量

* **shuffle** (bool, 可选, 默认为 True) - 在 Epoch 之前需要随机化 "rows"。

* **batch_size** (int, 可选, 默认为 1000) - 一次处理的 "rows" 数量。

* **target_batch_size** (int, optional, default = 1000) - Number of "rows" to accumulate gradients over. Useful when many rows will not fit into memory but are needed for accurate estimation.

* **classification** (bool, optional, default = True) - If True, problem is classification, else regression.

* **ordinal** (bool, optional, default = True) - If True, problem is ordinal classification. Requires classification to be True.

* **balanced** (bool, optional, default = True) - If true, each class is weighted equally in optimization, otherwise weighted is done via support of each class. Requires classification to be True.

* **prerocess** (str, optional, default = 'zscore') - 'zscore' which refers to centering and normalizing data to unit variance or 'center' which only centers the data to 0 mean.

* **soft_grouping** (bool, optional, default = True) - If True, groups represent features that come from the same source. Used to encourage sparsity of groups and features within groups.

* **verbose** (int, optional, default = 0) - Controls the verbosity when fitting. Set to 0 for no printing 1 or higher for printing every verbose number of gradient steps.

* **device** (str, optional, default = 'cpu') - 'cpu' to run on CPU and 'cuda' to run on GPU. Runs much faster on GPU


**Requirement of `fit` FuncArgs**

* **X** (array-like, require) - The training input samples which shape = [n_samples, n_features]

* **y** (array-like, require) - The target values (class labels in classification, real numbers in regression) which shape = [n_samples].

* **groups** (array-like, optional, default = None) - Groups of columns that must be selected as a unit. e.g. [0, 0, 1, 2] specifies the first two columns are part of a group. Which shape is [n_features].

**Requirement of `get_selected_features` FuncArgs**

 For now, the `get_selected_features` function has no parameters.

