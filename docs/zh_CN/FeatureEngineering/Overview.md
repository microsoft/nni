# 特征工程

我们很高兴的宣布，基于 NNI 的特征工程工具发布了 Alpha 版本。该版本仍处于试验阶段，根据使用反馈会进行改进。 诚挚邀请您使用、反馈，或更多贡献。

当前支持以下特征选择器：
- [GradientFeatureSelector](./GradientFeatureSelector.md)
- [GBDTSelector](./GBDTSelector.md)


# 如何使用

```python
from nni.feature_engineering.gradient_selector import GradientFeatureSelector
# from nni.feature_engineering.gbdt_selector import GBDTSelector

# 读取数据
...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 初始化 Selector
fgs = GradientFeatureSelector(...)
# 拟合数据
fgs.fit(X_train, y_train)
# 获取重要的特征
# 此处会返回重要特征的索引。
print(fgs.get_selected_features(...))

...
```

使用内置 Selector 时，需要 `import` 对应的特征选择器，并 `initialize`。 可在 Selector 中调用 `fit` 函数来传入数据。 之后，可通过 `get_seleteced_features` 来获得重要的特征。 不同 Selector 的函数参数可能不同，在使用前需要先检查文档。

# 如何定制

NNI 内置了_最先进的_特征工程算法的 Selector。 NNI 也支持定制自己的特征 Selector。

如果要实现定制的特征 Selector，需要：

1. 继承基类 FeatureSelector
1. 实现 _fit_ 和 _get_selected_features_ 函数
1. 与 sklearn 集成 (可选)

示例如下：

**1. 继承基类 FeatureSelector**

```python
from nni.feature_engineering.feature_selector import FeatureSelector

class CustomizedSelector(FeatureSelector):
    def __init__(self, ...):
    ...
```

**2. 实现 _fit_ 和 _get_selected_features_ 函数**

```python
from nni.tuner import Tuner

from nni.feature_engineering.feature_selector import FeatureSelector

class CustomizedSelector(FeatureSelector):
    def __init__(self, ...):
    ...

    def fit(self, X, y, **kwargs):
        """
        将数据拟合到 FeatureSelector

        参数
        ------------
        X : numpy 矩阵
        训练输入样本，形状为 [n_samples, n_features]。
        y: array-like numpy matrix
        The target values (class labels in classification, real numbers in regression). Which shape is [n_samples].
        """
        self.X = X
        self.y = y
        ...

    def get_selected_features(self):
        """
        Get important feature

        Returns
        -------
        list :
        Return the index of the important feature.
        """
        ...
        return self.selected_features_

    ...
```

**3. Integrate with Sklearn**

`sklearn.pipeline.Pipeline` can connect models in series, such as feature selector, normalization, and classification/regression to form a typical machine learning problem workflow. The following step could help us to better integrate with sklearn, which means we could treat the customized feature selector as a mudule of the pipeline.

1. Inherit the calss _sklearn.base.BaseEstimator_
1. Implement _get_params_ and _set_params_ function in _BaseEstimator_
1. Inherit the class _sklearn.feature_selection.base.SelectorMixin_
1. Implement _get_support_, _transform_ and _inverse_transform_ Function in _SelectorMixin_

Here is an example:

**1. Inherit the BaseEstimator Class and its Function**

```python
from sklearn.base import BaseEstimator
from nni.feature_engineering.feature_selector import FeatureSelector

class CustomizedSelector(FeatureSelector, BaseEstimator):
    def __init__(self, ...):
    ...

    def get_params(self, ...):
        """
        Get parameters for this estimator.
        """
        params = self.__dict__
        params = {key: val for (key, val) in params.items()
        if not key.endswith('_')}
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        for param in params:
        if hasattr(self, param):
        setattr(self, param, params[param])
        return self

```

**2. Inherit the SelectorMixin Class and its Function**
```python
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin

from nni.feature_engineering.feature_selector import FeatureSelector

class CustomizedSelector(FeatureSelector, BaseEstimator):
    def __init__(self, ...):
        ...

    def get_params(self, ...):
        """
        Get parameters for this estimator.
        """
        params = self.__dict__
        params = {key: val for (key, val) in params.items()
        if not key.endswith('_')}
        return params

        def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        for param in params:
        if hasattr(self, param):
        setattr(self, param, params[param])
        return self

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool
        Default False. If True, the return value will be an array of integers, rather than a boolean mask.

        Returns
        -------
        list :
        returns support: An index that selects the retained features from a feature vector.
        If indices are False, this is a boolean array of shape [# input features], in which an element is True iff its corresponding feature is selected for retention.
        If indices are True, this is an integer array of shape [# output features] whose values
        are indices into the input feature vector.
        """
        ...
        return mask


    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array
        which shape is [n_samples, n_features]

        Returns
        -------
        X_r : array
        which shape is [n_samples, n_selected_features]
        The input samples with only the selected features.
        """
        ...
        return X_r


    def inverse_transform(self, X):
        """
        Reverse the transformation operation

        Parameters
        ----------
        X : array
        shape is [n_samples, n_selected_features]

        Returns
        -------
        X_r : array
        shape is [n_samples, n_original_features]
        """
        ...
        return X_r
```

After integrating with Sklearn, we could use the feature selector as follows:
```python
from sklearn.linear_model import LogisticRegression

# load data
...
X_train, y_train = ...

# build a ppipeline
pipeline = make_pipeline(XXXSelector(...), LogisticRegression())
pipeline = make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression())
pipeline.fit(X_train, y_train)

# score
print("Pipeline Score: ", pipeline.score(X_train, y_train))

```

# Benchmark

`Baseline` means without any feature selection, we directly pass the data to LogisticRegression. For this benchmark, we only use 10% data from the train as test data.

| Dataset       | Baseline | GradientFeatureSelector | TreeBasedClassifier | #Train     | #Feature  |
| ------------- | -------- | ----------------------- | ------------------- | ---------- | --------- |
| colon-cancer  | 0.7547   | 0.7368                  | 0.7223              | 62         | 2,000     |
| gisette       | 0.9725   | 0.89416                 | 0.9792              | 6,000      | 5,000     |
| avazu         | 0.8834   | N/A                     | N/A                 | 40,428,967 | 1,000,000 |
| rcv1          | 0.9644   | 0.7333                  | 0.9615              | 20,242     | 47,236    |
| news20.binary | 0.9208   | 0.6870                  | 0.9070              | 19,996     | 1,355,191 |
| real-sim      | 0.9681   | 0.7969                  | 0.9591              | 72,309     | 20,958    |

The benchmark could be download in [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)

