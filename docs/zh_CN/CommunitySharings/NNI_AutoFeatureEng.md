# 来自知乎的评论： <an open source project with highly reasonable design> - 作者 Garvin Li

本文由 NNI 用户在知乎论坛上发表。 在这篇文章中，Garvin 分享了在使用 NNI 进行自动特征工程方面的体验。 我们认为本文对于有兴趣使用 NNI 进行特征工程的用户非常有用。 经作者许可，将原始文章摘编如下。

**原文**: [如何看待微软最新发布的AutoML平台NNI？作者 Garvin Li](https://www.zhihu.com/question/297982959/answer/964961829?utm_source=wechat_session&utm_medium=social&utm_oi=28812108627968&from=singlemessage&isappinstalled=0)

## 01 AutoML概述

作者认为 AutoML 不光是调参，应该包含自动特征工程。AutoML 是一个系统化的体系，包括：自动特征工程（AutoFeatureEng）、自动调参（AutoTuning）、自动神经网络探索（NAS）等。

## 02 NNI 概述

NNI（(Neural Network Intelligence）是一个微软的开源 AutoML 工具包，通过自动而有效的方法来帮助用户设计并调优机器学习模型，神经网络架构，或复杂系统的参数。

链接：[ https://github.com/Microsoft/nni](https://github.com/Microsoft/nni)

我目前只学习了自动特征工程这一个模块，总体看微软的工具都有一个比较大的特点，技术可能不一定多新颖，但是设计都非常赞。 NNI 的 AutoFeatureENG 基本包含了用户对于 AutoFeatureENG 的一切幻想。在微软做 PD 应该挺幸福吧，底层的这些个框架的设计都极为合理。

## 03 细说NNI - AutoFeatureENG
> 本文使用了此项目： [https://github.com/SpongebBob/tabular_automl_NNI](https://github.com/SpongebBob/tabular_automl_NNI)。

新用户可以使用 NNI 轻松高效地进行 AutoFeatureENG。 使用是非常简单的，安装下文件中的 require，然后 pip install NNI。

![](https://github.com/JSong-Jia/Pic/blob/master/images/pic%201.jpg) NNI把 AutoFeatureENG 拆分成 exploration 和 selection 两个模块。 exploration 主要是特征衍生和交叉，selection 讲的是如何做特征筛选。

## 04 特征 Exploration

对于功能派生，NNI 提供了许多可自动生成新功能的操作，[列表](https://github.com/SpongebBob/tabular_automl_NNI/blob/master/AutoFEOp.md)如下：

**count**: Count encoding is based on replacing categories with their counts computed on the train set, also named frequency encoding.

**target**: Target encoding is based on encoding categorical variable values with the mean of target variable per value.

**embedding**: Regard features as sentences, generate vectors using *Word2Vec.*

**crosscout**: Count encoding on more than one-dimension, alike CTR (Click Through Rate).

**aggregete**: Decide the aggregation functions of the features, including min/max/mean/var.

**nunique**: Statistics of the number of unique features.

**histsta**: Statistics of feature buckets, like histogram statistics.

Search space could be defined in a **JSON file**: to define how specific features intersect, which two columns intersect and how features generate from corresponding columns.

![](https://github.com/JSong-Jia/Pic/blob/master/images/pic%202.jpg)

The picture shows us the procedure of defining search space. NNI provides count encoding for 1-order-op, as well as cross count encoding, aggerate statistics (min max var mean median nunique) for 2-order-op.

For example, we want to search the features which are a frequency encoding (valuecount) features on columns name {“C1”, ...,” C26”}, in the following way:

![](https://github.com/JSong-Jia/Pic/blob/master/images/pic%203.jpg)

we can define a cross frequency encoding (value count on cross dims) method on columns {"C1",...,"C26"} x {"C1",...,"C26"} in the following way:

![](https://github.com/JSong-Jia/Pic/blob/master/images/pic%204.jpg)

The purpose of Exploration is to generate new features. You can use **get_next_parameter** function to get received feature candidates of one trial.
> RECEIVED_PARAMS = nni.get_next_parameter()

## 05 Feature selection

To avoid feature explosion and overfitting, feature selection is necessary. In the feature selection of NNI-AutoFeatureENG, LightGBM (Light Gradient Boosting Machine), a gradient boosting framework developed by Microsoft, is mainly promoted.

![](https://github.com/JSong-Jia/Pic/blob/master/images/pic%205.jpg)

If you have used **XGBoost** or **GBDT**, you would know the algorithm based on tree structure can easily calculate the importance of each feature on results. LightGBM is able to make feature selection naturally.

The issue is that selected features might be applicable to *GBDT* (Gradient Boosting Decision Tree), but not to the linear algorithm like *LR* (Logistic Regression).

![](https://github.com/JSong-Jia/Pic/blob/master/images/pic%206.jpg)

## 06 Summary

NNI's AutoFeatureEng sets a well-established standard, showing us the operation procedure, available modules, which is highly convenient to use. However, a simple model is probably not enough for good results.

## Suggestions to NNI

About Exploration: If consider using DNN (like xDeepFM) to extract high-order feature would be better.

About Selection: There could be more intelligent options, such as automatic selection system based on downstream models.

Conclusion: NNI could offer users some inspirations of design and it is a good open source project. I suggest researchers leverage it to accelerate the AI research.

Tips: Because the scripts of open source projects are compiled based on gcc7, Mac system may encounter problems of gcc (GNU Compiler Collection). The solution is as follows:

# brew install libomp

