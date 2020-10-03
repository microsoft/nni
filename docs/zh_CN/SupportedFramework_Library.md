# 框架和库的支持

通过内置的 Python API，NNI 天然支持所有 Python (` 版本 >= 3.6`) 语言的 AI 框架，可使用所有超参调优和神经网络搜索算法。 NNI 还为常见场景提供了一些示例和教程，使上手更容易。

## 支持的 AI 框架

* **[PyTorch]** https://github.com/pytorch/pytorch

* [MNIST-pytorch](../../examples/trials/mnist-distributed-pytorch)  
    
* [CIFAR-10](TrialExample/Cifar10Examples.md)  
    
* [TGS salt identification chanllenge](../../examples/trials/kaggle-tgs-salt/README_zh_CN.md)  
    
* [Network morphism](../../examples/trials/network_morphism/README_zh_CN.md)  
    

* **[TensorFlow]** https://github.com/tensorflow/tensorflow

* [MNIST-tensorflow](../../examples/trials/mnist-distributed)  
    
* [Squad](../../examples/trials/ga_squad/README_zh_CN.md)  
    

* **[Keras]** https://github.com/keras-team/keras

* [MNIST-keras](../../examples/trials/mnist-keras)  
    
* [Network morphism](../../examples/trials/network_morphism/README_zh_CN.md)  
    

* **[MXNet]** https://github.com/apache/incubator-mxnet
* **[Caffe2]** https://github.com/BVLC/caffe
* **[CNTK (Python 语言)]** https://github.com/microsoft/CNTK
* **[Spark MLlib]** http://spark.apache.org/mllib/
* **[Chainer]** https://chainer.org/
* **[Theano]** https://pypi.org/project/Theano/   
    

如果能[贡献更多示例](Tutorial/Contributing.md)，会对其他 NNI 用户有很大的帮助。

## 支持的库

NNI 也支持其它 Python 库，包括一些基于 GBDT 的算法：XGBoost, CatBoost 以及 lightGBM。

* **[Scikit-learn]** https://scikit-learn.org/stable/

* [Scikit-learn](TrialExample/SklearnExamples.md)  
    

* **[XGBoost]** https://xgboost.readthedocs.io/en/latest/
* **[CatBoost]** https://catboost.ai/
* **[LightGBM]** https://lightgbm.readthedocs.io/en/latest/

* [Auto-gbdt](TrialExample/GbdtExample.md)  
    

这只是 NNI 支持的一小部分库。 如果对 NNI 感兴趣，可参考[教程](TrialExample/Trials.md)来继续学习。

除了这些案例，也欢迎更多的用户将 NNI 应用到自己的工作中，如果有任何疑问，请参考[实现 Trial](TrialExample/Trials.md)。 如果想成为 NNI 的贡献者，无论是分享示例，还是实现 Tuner 或其它内容，我们都非常期待您的参与。更多信息请[参考这里](Tutorial/Contributing.md)。