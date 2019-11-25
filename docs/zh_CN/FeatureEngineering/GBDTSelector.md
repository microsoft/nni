## GBDTSelector

GBDTSelector 基于 [LightGBM](https://github.com/microsoft/LightGBM)，这是一个基于树学习算法的梯度提升框架。

当将数据传递到 GBDT 模型时，该模型将构建提升树。 特征的重要性来自于构造时的分数，其表达了每个特征在模型构造提升决策树时有多有用。

可使用此方法作为 Feature Selector 中较强的基准，特别是在使用 GBDT 模型进行分类或回归时。

当前，支持的 `importance_type` 有 `split` 和 `gain`。 未来会支持定制 `importance_type`，也就是说用户可以定义如何计算`特征分数`。

### 用法

首先，安装依赖项：

```
pip install lightgbm
```

然后

```python
from nni.feature_engineering.gbdt_selector import GBDTSelector

# 读取数据
...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# initlize a selector
fgs = GBDTSelector()
# fit data
fgs.fit(X_train, y_train, ...)
# get improtant features
# will return the index with important feature here.
print(fgs.get_selected_features(10))

...
```

And you could reference the examples in `/examples/feature_engineering/gbdt_selector/`, too.


**Requirement of `fit` FuncArgs**

* **X** (array-like, require) - The training input samples which shape = [n_samples, n_features]

* **y** (array-like, require) - The target values (class labels in classification, real numbers in regression) which shape = [n_samples].

* **lgb_params** (dict, require) - The parameters for lightgbm model. The detail you could reference [here](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

* **eval_ratio** (float, require) - The ratio of data size. It's used for split the eval data and train data from self.X.

* **early_stopping_rounds** (int, require) - The early stopping setting in lightgbm. The detail you could reference [here](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

* **importance_type** (str, require) - could be 'split' or 'gain'. The 'split' means ' result contains numbers of times the feature is used in a model' and the 'gain' means 'result contains total gains of splits which use the feature'. The detail you could reference in [here](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance).

* **num_boost_round** (int, require) - number of boost round. The detail you could reference [here](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm.train).

**Requirement of `get_selected_features` FuncArgs**

* **topk** (int, require) - the topK impotance features you want to selected.

