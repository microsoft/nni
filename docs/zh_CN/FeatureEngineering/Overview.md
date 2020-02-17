# NNI 中的特征工程

我们很高兴的宣布，基于 NNI 的特征工程工具发布了试用版本。该版本仍处于试验阶段，根据使用反馈会进行改进。 诚挚邀请您使用、反馈，或更多贡献。

当前支持以下特征选择器：
- [GradientFeatureSelector](./GradientFeatureSelector.md)
- [GBDTSelector](./GBDTSelector.md)

These selectors are suitable for tabular data(which means it doesn't include image, speech and text data).

In addition, those selector only for feature selection. If you want to: 1) generate high-order combined features on nni while doing feature selection; 2) leverage your distributed resources; you could try this [example](https://github.com/microsoft/nni/tree/master/examples/feature_engineering/auto-feature-engineering).

## 如何使用

```python
from nni.feature_engineering.gradient_selector import FeatureGradientSelector
# from nni.feature_engineering.gbdt_selector import GBDTSelector

# 读取数据
...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 初始化 Selector
fgs = FeatureGradientSelector(...)
# 拟合数据
fgs.fit(X_train, y_train)
# 获取重要的特征
# 此处会返回重要特征的索引。
print(fgs.get_selected_features(...))

...
```

When using the built-in Selector, you first need to `import` a feature selector, and `initialize` it. You could call the function `fit` in the selector to pass the data to the selector. After that, you could use `get_seleteced_features` to get important features. The function parameters in different selectors might be different, so you need to check the docs before using it.

## 如何定制？

NNI provides _state-of-the-art_ feature selector algorithm in the builtin-selector. NNI also supports to build a feature selector by yourself.

If you want to implement a customized feature selector, you need to:

1. 继承基类 FeatureSelector
1. 实现 _fit_ 和 _get_selected_features_ 函数
1. 与 sklearn 集成 (可选)

Here is an example:

**1. Inherit the base Featureselector Class**

```python
from nni.feature_engineering.feature_selector import FeatureSelector

class CustomizedSelector(FeatureSelector):
    def __init__(self, ...):
    ...
```

**2. Implement _fit_ and _get_selected_features_ Function**

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
        y: numpy 矩阵
        目标值 (分类中的类标签，回归中为实数)。 形状是 [n_samples]。
        """
        self.X = X
        self.y = y
        ...

    def get_selected_features(self):
        """
        获取重要特征

        Returns
        -------
        list :
        返回重要特征的索引。
        """
        ...
        return self.selected_features_

    ...
```

**3. Integrate with Sklearn**

`sklearn.pipeline.Pipeline` can connect models in series, such as feature selector, normalization, and classification/regression to form a typical machine learning problem workflow. The following step could help us to better integrate with sklearn, which means we could treat the customized feature selector as a mudule of the pipeline.

1. 继承类 _sklearn.base.BaseEstimator_
1. 实现 _BaseEstimator_ 中的 _get_params_ 和 _set_params_ 函数
1. 继承类 _sklearn.feature_selection.base.SelectorMixin_
1. 实现 _SelectorMixin_ 中的 _get_support_, _transform_ 和 _inverse_transform_ 函数

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
        为此 estimator 获取参数
        """
        params = self.__dict__
        params = {key: val for (key, val) in params.items()
        if not key.endswith('_')}
        return params

    def set_params(self, **params):
        """
        为此 estimator 设置参数
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
        获取参数。
        """
        params = self.__dict__
        params = {key: val for (key, val) in params.items()
        if not key.endswith('_')}
        return params

        def set_params(self, **params):
        """
        设置参数
        """
        for param in params:
        if hasattr(self, param):
        setattr(self, param, params[param])
        return self

    def get_support(self, indices=False):
        """
        获取 mask，整数索引或选择的特征。

        Parameters
        ----------
        indices : bool
        默认为 False. 如果为 True，返回值为整数数组，否则为布尔的 mask。

        Returns
        -------
        list :
        返回 support: 从特征向量中选择保留的特征索引。
        如果 indices 为 False，布尔数据的形状为 [输入特征的数量]，如果元素为 True，表示保留相对应的特征。
        如果 indices 为 True，整数数组的形状为 [输出特征的数量]，值表示
        输入特征向量中的索引。
        """
        ...
        return mask


    def transform(self, X):
        """将 X 减少为选择的特征。

        Parameters
        ----------
        X : array
        形状为 [n_samples, n_features]

        Returns
        -------
        X_r : array
        形状为 [n_samples, n_selected_features]
        仅输入选择的特征。
        """
        ...
        return X_r


    def inverse_transform(self, X):
        """
        反转变换操作

        Parameters
        ----------
        X : array
        形状为 [n_samples, n_selected_features]

        Returns
        -------
        X_r : array
        形状为 [n_samples, n_original_features]
        """
        ...
        return X_r
```

After integrating with Sklearn, we could use the feature selector as follows:
```python
from sklearn.linear_model import LogisticRegression

# 加载数据
...
X_train, y_train = ...

# 构造 pipeline
pipeline = make_pipeline(XXXSelector(...), LogisticRegression())
pipeline = make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression())
pipeline.fit(X_train, y_train)

# 分数
print("Pipeline Score: ", pipeline.score(X_train, y_train))

```

## 基准测试

`Baseline` means without any feature selection, we directly pass the data to LogisticRegression. For this benchmark, we only use 10% data from the train as test data. For the GradientFeatureSelector, we only take the top20 features. The metric is the mean accuracy on the given test data and labels.

| 数据集           | 所有特征 + LR (acc, time, memory) | GradientFeatureSelector + LR (acc, time, memory) | TreeBasedClassifier + LR (acc, time, memory) | 训练次数       | 特征数量      |
| ------------- | ----------------------------- | ------------------------------------------------ | -------------------------------------------- | ---------- | --------- |
| colon-cancer  | 0.7547, 890ms, 348MiB         | 0.7368, 363ms, 286MiB                            | 0.7223, 171ms, 1171 MiB                      | 62         | 2,000     |
| gisette       | 0.9725, 215ms, 584MiB         | 0.89416, 446ms, 397MiB                           | 0.9792, 911ms, 234MiB                        | 6,000      | 5,000     |
| avazu         | 0.8834, N/A, N/A              | N/A, N/A, N/A                                    | N/A, N/A, N/A                                | 40,428,967 | 1,000,000 |
| rcv1          | 0.9644, 557ms, 241MiB         | 0.7333, 401ms, 281MiB                            | 0.9615, 752ms, 284MiB                        | 20,242     | 47,236    |
| news20.binary | 0.9208, 707ms, 361MiB         | 0.6870, 565ms, 371MiB                            | 0.9070, 904ms, 364MiB                        | 19,996     | 1,355,191 |
| real-sim      | 0.9681, 433ms, 274MiB         | 0.7969, 251ms, 274MiB                            | 0.9591, 643ms, 367MiB                        | 72,309     | 20,958    |

The dataset of benchmark could be download in [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)

The code could be refenrence `/examples/feature_engineering/gradient_feature_selector/benchmark_test.py`.

## 参考和反馈
* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)；
* 了解 NNI 中[神经网络结构搜索的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/NAS/Overview.md)；
* 了解 NNI 中[模型自动压缩的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Compressor/Overview.md)；
* 了解如何[使用 NNI 进行超参数调优](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Tuner/BuiltinTuner.md)；
