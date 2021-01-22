.. role:: raw-html(raw)
   :format: html


Framework and Library Supports
==============================

With the built-in Python API, NNI naturally supports the hyper parameter tuning and neural network search for all the AI frameworks and libraries who support Python models(\ ``version >= 3.6``\ ). NNI had also provided a set of examples and tutorials for some of the popular scenarios to make jump start easier.

Supported AI Frameworks
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

You are encouraged to `contribute more examples <Tutorial/Contributing.rst>`__ for other NNI users. 

Supported Library
-----------------

NNI also supports all libraries written in python.Here are some common libraries, including some algorithms based on GBDT: XGBoost, CatBoost and lightGBM.


* `Scikit-learn <https://scikit-learn.org/stable/>`__

  * `Scikit-learn <TrialExample/SklearnExamples.rst>`__

* `XGBoost <https://xgboost.readthedocs.io/en/latest/>`__
* `CatBoost <https://catboost.ai/>`__
* `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`__

  * `Auto-gbdt <TrialExample/GbdtExample.rst>`__

Here is just a small list of libraries that supported by NNI. If you are interested in NNI, you can refer to the `tutorial <TrialExample/Trials.rst>`__ to complete your own hacks.

In addition to the above examples, we also welcome more and more users to apply NNI to your own work, if you have any doubts, please refer `Write a Trial Run on NNI <TrialExample/Trials.rst>`__. In particular, if you want to be a contributor of NNI, whether it is the sharing of examples , writing of Tuner or otherwise, we are all looking forward to your participation.More information please refer to `here <Tutorial/Contributing.rst>`__.
