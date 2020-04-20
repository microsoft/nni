# GBDT

梯度提升是机器学习中回归和分类问题的一种方法。它由一组弱分类模型所组成，决策树是其中的典型。 像其它提升方法一样，它也分步来构建模型，并使用可微分的损失函数来优化。

梯度决策树（gradient boosting decision tree，GBDT）有很多流行的实现，如：[LightGBM](https://github.com/Microsoft/LightGBM), [xgboost](https://github.com/dmlc/xgboost), 和 [catboost](https://github.com/catboost/catboost)，等等。 GBDT 是解决经典机器学习问题的重要工具。 GBDT 也是一种鲁棒的算法，可以使用在很多领域。 GBDT 的超参越好，就能获得越好的性能。

NNI 是用来调优超参的平台，可以在 NNI 中尝试各种内置的搜索算法，并行运行多个 Trial。

## 1. GBDT 的搜索空间

GBDT 有很多超参，但哪些才会影响性能或计算速度呢？ 基于实践经验，建议如下（以 lightgbm 为例）：

> * 获得更好的精度

* `learning_rate`. `学习率`的范围应该是 [0.001, 0.9]。

* `num_leaves`. `num_leaves` 与 `max_depth` 有关，不必两个值同时调整。

* `bagging_freq`. `bagging_freq` 可以是 [1, 2, 4, 8, 10]。

* `num_iterations`. 如果达到期望的拟合精度，可以调整得大一些。

> * 加速

* `bagging_fraction`. `bagging_fraction` 的范围应该是 [0.7, 1.0]。

* `feature_fraction`. `feature_fraction` 的范围应该是 [0.6, 1.0]。

* `max_bin`.

> * 避免过拟合

* `min_data_in_leaf`. 取决于数据集。

* `min_sum_hessian_in_leaf`. 取决于数据集。

* `lambda_l1` 和 `lambda_l2`.

* `min_gain_to_split`.

* `num_leaves`.

更多信息可参考： [lightgbm](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) 和 [autoxgoboost](https://github.com/ja-thomas/autoxgboost/blob/master/poster_2018.pdf)

## 2. 任务描述

"auto-gbdt" 基于 LightGBM 和 NNI。 数据集有[训练数据](https://github.com/Microsoft/nni/blob/master/examples/trials/auto-gbdt/data/regression.train)和[测试数据](https://github.com/Microsoft/nni/blob/master/examples/trials/auto-gbdt/data/regression.train)。 根据数据中的特征和标签，训练一个 GBDT 回归模型，用来做预测。

## 3. 如何运行 NNI

### 3.1 安装所有要求的包

    pip install lightgbm
    pip install pandas
    

### 3.2 准备 Trial 代码

基础代码如下：

```python
...

def get_default_parameters():
    ...
    return params


def load_data(train_path='./data/regression.train', test_path='./data/regression.test'):
    '''
    读取或创建数据集
    '''
    ...

    return lgb_train, lgb_eval, X_test, y_test

def run(lgb_train, lgb_eval, params, X_test, y_test):
    # 训练
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)
    # 预测
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # 评估
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print('The rmse of prediction is:', rmse)

if __name__ == '__main__':
    lgb_train, lgb_eval, X_test, y_test = load_data()

    PARAMS = get_default_parameters()
    # train
    run(lgb_train, lgb_eval, PARAMS, X_test, y_test)
```

### 3.3 准备搜索空间

如果要调优 `num_leaves`, `learning_rate`, `bagging_fraction` 和 `bagging_freq`, 可创建一个 [search_space.json](https://github.com/Microsoft/nni/blob/master/examples/trials/auto-gbdt/search_space.json) 文件：

```json
{
    "num_leaves":{"_type":"choice","_value":[31, 28, 24, 20]},
    "learning_rate":{"_type":"choice","_value":[0.01, 0.05, 0.1, 0.2]},
    "bagging_fraction":{"_type":"uniform","_value":[0.7, 1.0]},
    "bagging_freq":{"_type":"choice","_value":[1, 2, 4, 8, 10]}
}
```

参考[这里](../Tutorial/SearchSpaceSpec.md)，了解更多变量类型。

### 3.4 在代码中使用 NNI SDK

```diff
+import nni
...

def get_default_parameters():
    ...
    return params


def load_data(train_path='./data/regression.train', test_path='./data/regression.test'):
    '''
    读取或创建数据集
    '''
    ...

    return lgb_train, lgb_eval, X_test, y_test

def run(lgb_train, lgb_eval, params, X_test, y_test):
    # 训练
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)
    # 预测
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # 评估
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print('The rmse of prediction is:', rmse)

+   nni.report_final_result(rmse)

if __name__ == '__main__':
    lgb_train, lgb_eval, X_test, y_test = load_data()

+   RECEIVED_PARAMS = nni.get_next_parameter()
    PARAMS = get_default_parameters()
+   PARAMS.update(RECEIVED_PARAMS)

    # 训练
    run(lgb_train, lgb_eval, PARAMS, X_test, y_test)
```

### 3.5 实现配置文件并运行

在配置文件中，可以设置如下内容：

* Experiment 设置：`trialConcurrency`, `maxExecDuration`, `maxTrialNum`, `trial gpuNum`, 等等。
* 平台设置：`trainingServicePlatform`，等等。
* 路径设置：`searchSpacePath`, `trial codeDir`，等等。
* 算法设置：选择 `Tuner` 算法，`优化方向`，等等。

config.yml 示例：

```yaml
authorName: default
experimentName: example_auto-gbdt
trialConcurrency: 1
maxExecDuration: 10h
maxTrialNum: 10
#可选项: local, remote, pai
trainingServicePlatform: local
searchSpacePath: search_space.json
#可选项: true, false
useAnnotation: false
tuner:
  #可选项: TPE, Random, Anneal, Evolution, BatchTuner
  #SMAC (SMAC 需要先通过 nnictl 来安装)
  builtinTunerName: TPE
  classArgs:
    #可选项: maximize, minimize
    optimize_mode: minimize
trial:
  command: python3 main.py
  codeDir: .
  gpuNum: 0
```

使用下面的命令启动 Experiment：

```bash
nnictl create --config ./config.yml
```