# NNI 中使用 scikit-learn

[scikit-learn](https://github.com/scikit-learn/scikit-learn) (sklearn) 是数据挖掘和分析的流行工具。 它支持多种机器学习模型，如线性回归，逻辑回归，决策树，支持向量机等。 提高 scikit-learn 的效率是非常有价值的课题。  
NNI 支持多种调优算法，可以为 scikit-learn 搜索最佳的模型和超参，并支持本机、远程服务器组、云等各种环境。

## 1. 如何运行此样例

安装 NNI 包，并使用命令行工具 `nnictl` 来启动 Experiment。 有关安装和环境准备的内容，参考[这里](QuickStart.md)。 安装完 NNI 后，进入相应的目录，输入下列命令即可启动 Experiment：

```bash
nnictl create --config ./config.yml
```

## 2. 样例概述

### 2.1 分类

此样例使用了数字数据集，由 1797 张 8x8 的图片组成，每张图片都是一个手写数字。目标是将这些图片分到 10 个类别中。  
在此样例中，使用了 SVC 作为模型，并选择了一些参数，包括 `"C", "keral", "degree", "gamma" 和 "coef0"`。 关于这些参数的更多信息，可参考[这里](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)。

### 2.2 回归

此样例使用了波士顿房价数据，数据集由波士顿各地区房价所组成，还包括了房屋的周边信息，例如：犯罪率 (CRIM)，非零售业务的面积 (INDUS)，房主年龄 (AGE) 等等。这些信息可用来预测波士顿的房价。 本例中，尝试了不同的回归模型，包括 `"LinearRegression", "SVR", "KNeighborsRegressor", "DecisionTreeRegressor"` 和一些参数，如 `"svr_kernel", "knr_weights"`。 关于这些模型算法和参数的更多信息，可参考[这里](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)。

## 3. 如何在 NNI 中使用 sklearn

只需要如下几步，即可在 sklearn 代码中使用 NNI。

* **第一步**
    
    准备 search_space.json 文件来存储选择的搜索空间。 例如，如果要在不同的模型中选择：
    
    ```json
    {
    "model_name":{"_type":"choice","_value":["LinearRegression", "SVR", "KNeighborsRegressor", "DecisionTreeRegressor"]}
    }
    ```
    
    If you want to choose different models and parameters, you could put them together in a search_space.json file.
    
    ```json
    {
    "model_name":{"_type":"choice","_value":["LinearRegression", "SVR", "KNeighborsRegressor", "DecisionTreeRegressor"]},
    "svr_kernel": {"_type":"choice","_value":["linear", "poly", "rbf"]},
    "knr_weights": {"_type":"choice","_value":["uniform", "distance"]}
    }
    ```
    
    Then you could read these values as a dict from your python code, please get into the step 2.

* **step 2**  
    At the beginning of your python code, you should `import nni` to insure the packages works normally. First, you should use `nni.get_next_parameter()` function to get your parameters given by nni. Then you could use these parameters to update your code. For example, if you define your search_space.json like following format:
    
    ```json
    {
    "C": {"_type":"uniform","_value":[0.1, 1]},
    "keral": {"_type":"choice","_value":["linear", "rbf", "poly", "sigmoid"]},
    "degree": {"_type":"choice","_value":[1, 2, 3, 4]},
    "gamma": {"_type":"uniform","_value":[0.01, 0.1]},
    "coef0 ": {"_type":"uniform","_value":[0.01, 0.1]}
    }
    ```
    
    You may get a parameter dict like this:
    
    ```python
    params = {
        'C': 1.0,
        'keral': 'linear',
        'degree': 3,
        'gamma': 0.01,
        'coef0': 0.01
    }
    ```
    
    Then you could use these variables to write your scikit-learn code.

* **step 3**  
    After you finished your training, you could get your own score of the model, like your percision, recall or MSE etc. NNI needs your score to tuner algorithms and generate next group of parameters, please report the score back to NNI and start next trial job.  
    You just need to use `nni.report_final_result(score)` to communitate with NNI after you process your scikit-learn code. Or if you have multiple scores in the steps of training, you could also report them back to NNI using `nni.report_intemediate_result(score)`. Note, you may not report intemediate result of your job, but you must report back your final result.