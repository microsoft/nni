## GradientFeatureSelector

GradientFeatureSelector 的算法来源于 ["Feature Gradients: Scalable Feature Selection via Discrete Relaxation"](https://arxiv.org/pdf/1908.10382.pdf)。

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

* **target_batch_size** (int, 可选, 默认为 1000) - 累计梯度的 "rows" 数量。 当行数过多无法读取到内存中，但估计精度所需。

* **classification** (bool, 可选, 默认为 True) - 如果为 True，为分类问题，否则是回归问题。

* **ordinal** (bool, 可选, 默认为 True) - 如果为 True，是有序的分类。 需要 classification 也为 True。

* **balanced** (bool, 可选, 默认为 True) - 如果为 True，优化中每类的权重都一样，否则需要通过支持来对每类加权。 需要 classification 也为 True。

* **prerocess** (str, 可选, 默认为 'zscore') - 'zscore' 是将数据中心化并归一化的党委方差，'center' 表示仅将数据均值调整到 0。

* **soft_grouping** (bool, 可选, 默认为 True) - 如果为 True，将同一来源的特征分组到一起。 用于支持分组或组内特征的稀疏性。

* **verbose** (int, 可选, 默认为 0) - 控制拟合时的信息详细程度。 设为 0 表示不打印，1 或更大值表示打印详细数量的步骤。

* **device** (str, 可选, 默认为 'cpu') - 'cpu' 表示在 CPU 上运行，'cuda' 表示在 GPU 上运行。 在 GPU 上运行得更快


**`fit` 函数参数要求**

* **X** (数组，必需) - 训练的输入样本，shape = [n_samples, n_features]

* **y** (数组，必需) - 目标值 (分类中为标签，回归中为实数)，shape = [n_samples].

* **groups** (数组, 可选, 默认为 None) - 必需选择为一个单元的列的分组。 例如 [0，0，1，2] 指定前两列是组的一部分。 形状是 [n_features]。

**`get_selected_features` 函数参数的要求**

 目前， `get_selected_features` 函数没有参数。

