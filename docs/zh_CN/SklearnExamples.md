# NNI 中使用 scikit-learn

[Scikit-learn](https://github.com/scikit-learn/scikit-learn) is a popular machine learning tool for data mining and data analysis. It supports many kinds of machine learning models like LinearRegression, LogisticRegression, DecisionTree, SVM etc. 如何更高效的使用 scikit-learn，是一个很有价值的话题。

NNI supports many kinds of tuning algorithms to search the best models and/or hyper-parameters for scikit-learn, and support many kinds of environments like local machine, remote servers and cloud.

## 1. 如何运行此样例

To start using NNI, you should install the NNI package, and use the command line tool `nnictl` to start an experiment. For more information about installation and preparing for the environment, please refer [here](QuickStart.md).

After you installed NNI, you could enter the corresponding folder and start the experiment using following commands:

```bash
nnictl create --config ./config.yml
```

## 2. 样例概述

### 2.1 分类

This example uses the dataset of digits, which is made up of 1797 8x8 images, and each image is a hand-written digit, the goal is to classify these images into 10 classes.

In this example, we use SVC as the model, and choose some parameters of this model, including `"C", "keral", "degree", "gamma" and "coef0"`. For more information of these parameters, please [refer](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

### 2.2 回归

This example uses the Boston Housing Dataset, this dataset consists of price of houses in various places in Boston and the information such as Crime (CRIM), areas of non-retail business in the town (INDUS), the age of people who own the house (AGE) etc., to predict the house price of Boston.

In this example, we tune different kinds of regression models including `"LinearRegression", "SVR", "KNeighborsRegressor", "DecisionTreeRegressor"` and some parameters like `"svr_kernel", "knr_weights"`. You could get more details about these models from [here](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning).

## 3. How to write scikit-learn code using NNI

It is easy to use NNI in your scikit-learn code, there are only a few steps.

* **第一步**
    
    准备 search_space.json 文件来存储选择的搜索空间。 例如，如果要在不同的模型中选择：
    
    ```json
    {
    "model_name":{"_type":"choice","_value":["LinearRegression", "SVR", "KNeighborsRegressor", "DecisionTreeRegressor"]}
    }
    ```
    
    如果要选择不同的模型和参数，可以将它们放到同一个 search_space.json 文件中。
    
    ```json
    {
    "model_name":{"_type":"choice","_value":["LinearRegression", "SVR", "KNeighborsRegressor", "DecisionTreeRegressor"]},
    "svr_kernel": {"_type":"choice","_value":["linear", "poly", "rbf"]},
    "knr_weights": {"_type":"choice","_value":["uniform", "distance"]}
    }
    ```
    
    在 Python 代码中，可以将这些值作为一个 dict，读取到 Python 代码中。

* **step 2**
    
    At the beginning of your python code, you should `import nni` to insure the packages works normally.
    
    First, you should use `nni.get_next_parameter()` function to get your parameters given by NNI. Then you could use these parameters to update your code. For example, if you define your search_space.json like following format:
    
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
    
    After you finished your training, you could get your own score of the model, like your precision, recall or MSE etc. NNI needs your score to tuner algorithms and generate next group of parameters, please report the score back to NNI and start next trial job.
    
    You just need to use `nni.report_final_result(score)` to communicate with NNI after you process your scikit-learn code. Or if you have multiple scores in the steps of training, you could also report them back to NNI using `nni.report_intemediate_result(score)`. Note, you may not report intermediate result of your job, but you must report back your final result.