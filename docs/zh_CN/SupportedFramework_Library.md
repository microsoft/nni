# 框架和库的支持

通过内置的 Python API，NNI 天然支持所有 Python (` 版本 >= 3.5`) 语言的 AI 框架，可使用所有超参调优和神经网络搜索算法。 NNI 还为常见场景提供了一些示例和教程，使上手更容易。

## 支持的 AI 框架

* **[PyTorch]** https://github.com/pytorch/pytorch

* [MNIST-pytorch](../../examples/trials/mnist-distributed-pytorch)  
    
* [CIFAR-10](TrialExample/Cifar10Examples.md)  
    
* [TGS salt identification chanllenge](../../examples/trials/kaggle-tgs-salt/README.md)  
    
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
    

You are encouraged to [contribute more examples](Tutorial/Contributing.md) for other NNI users.

## Supported Library

NNI also supports all libraries written in python.Here are some common libraries, including some algorithms based on GBDT: XGBoost, CatBoost and lightGBM.

* **[Scikit-learn]** https://scikit-learn.org/stable/

* [Scikit-learn](TrialExample/SklearnExamples.md)  
    

* **[XGBoost]** https://xgboost.readthedocs.io/en/latest/
* **[CatBoost]** https://catboost.ai/
* **[LightGBM]** https://lightgbm.readthedocs.io/en/latest/

* [Auto-gbdt](TrialExample/GbdtExample.md)  
    

Here is just a small list of libraries that supported by NNI. If you are interested in NNI, you can refer to the [tutorial](TrialExample/Trials.md) to complete your own hacks.

In addition to the above examples, we also welcome more and more users to apply NNI to your own work, if you have any doubts, please refer [Write a Trial Run on NNI](TrialExample/Trials.md). In particular, if you want to be a contributor of NNI, whether it is the sharing of examples , writing of Tuner or otherwise, we are all looking forward to your participation.More information please refer to [here](Tutorial/Contributing.md).