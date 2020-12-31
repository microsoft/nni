.. role:: raw-html(raw)
   :format: html


框架和库的支持
==============================

通过内置的 Python API，NNI 天然支持所有 Python（ ``版本 >= 3.6`` ）语言的 AI 框架，可使用所有超参调优和神经网络搜索算法。 NNI 还为常见场景提供了一些示例和教程，使上手更容易。

支持的 AI 框架
-----------------------


* `PyTorch <https://github.com/pytorch/pytorch>`__

  * :githublink:`MNIST-pytorch <examples/trials/mnist-distributed-pytorch>`
  * `CIFAR-10 <./TrialExample/Cifar10Examples.rst>`__
  * :githublink:`TGS salt identification chanllenge <examples/trials/kaggle-tgs-salt/README.md>`
  * :githublink:`Network_morphism <examples/trials/network_morphism/README.md>`

* `TensorFlow <https://github.com/tensorflow/tensorflow>`__

  * :githublink:`MNIST-tensorflow <examples/trials/mnist-distributed>`
  * :githublink:`Squad <examples/trials/ga_squad/README.md>`

* `Keras <https://github.com/keras-team/keras>`__

  * :githublink:`MNIST-keras <examples/trials/mnist-keras>`
  * :githublink:`Network_morphism <examples/trials/network_morphism/README.md>`


* `MXNet <https://github.com/apache/incubator-mxnet>`__
* `Caffe2 <https://github.com/BVLC/caffe>`__
* `CNTK (Python language) <https://github.com/microsoft/CNTK>`__
* `Spark MLlib <http://spark.apache.org/mllib/>`__
* `Chainer <https://chainer.org/>`__
* `Theano <https://pypi.org/project/Theano/>`__

鼓励您为其他的 NNI 用户\ `贡献更多示例 <Tutorial/Contributing.rst>`__  

支持的库
-----------------

NNI 也支持其它 Python 库，包括一些基于 GBDT 的算法：XGBoost, CatBoost 以及 lightGBM。

* `Scikit-learn <https://scikit-learn.org/stable/>`__

  * `Scikit-learn <TrialExample/SklearnExamples.rst>`__

* `XGBoost <https://xgboost.readthedocs.io/en/latest/>`__
* `CatBoost <https://catboost.ai/>`__
* `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`__

  * `Auto-gbdt <TrialExample/GbdtExample.rst>`__

这只是 NNI 支持的一小部分库。 如果对 NNI 感兴趣，可参考 `此教程 <TrialExample/Trials.rst>`__ 来继续学习。

除了这些案例，也欢迎更多的用户将 NNI 应用到自己的工作中，如果有任何疑问，请参考 `在 NNI 上实现 Trial <TrialExample/Trials.rst>`__ 。 如果想成为 NNI 的贡献者，无论是分享示例，还是实现 Tuner 或其它内容，我们都非常期待您的参与。更多信息请参考 `这里 <Tutorial/Contributing.rst>`__ 。
