# Framework and Library Supports 
With the built-in Python API, NNI naturally supports the hyper parameter tuning and neural network search for all the AI frameworks and libraries who support Python models(`version >= 3.5`). NNI had also provided a set of examples and tutorials for some of the popular scenarios to make jump start easier.

## Supported AI Frameworks:

* [PyTorch]https://github.com/pytorch/pytorch
    <ul> 
      <li><a href="../../examples/trials/mnist-distributed-pytorch">MNIST-pytorch</a><br/></li>
      <li><a href="TrialExample/Cifar10Examples.md">CIFAR-10</a><br/></li>
      <li><a href="../../examples/trials/kaggle-tgs-salt/README.md">TGS salt identification chanllenge</a><br/></li>
      <li><a href="../../examples/trials/network_morphism/README.md">Network_morphism</a><br/></li>
    </ul>
* [TensorFlow]https://github.com/tensorflow/tensorflow
    <ul> 
      <li><a href="../../examples/trials/mnist-distributed">MNIST-tensorflow</a><br/></li>
       <li><a href="../../examples/trials/ga_squad/README.md">Squad</a><br/></li>
    </ul>
* [Keras]https://github.com/keras-team/keras
    <ul>
      <li><a href="../../examples/trials/mnist-keras">MNIST-keras</a><br/></li>
      <li><a href="../../examples/trials/network_morphism/README.md">Network_morphism</a><br/></li>
    </ul>
* [MXNet]https://github.com/apache/incubator-mxnet
* [Caffe2]https://github.com/BVLC/caffe
* [CNTK (Python language)]https://github.com/microsoft/CNTK
* [Spark MLlib]http://spark.apache.org/mllib/
* [Chainer]https://chainer.org/
* [Theano]https://pypi.org/project/Theano/ <br/>

You are encouraged to [contribute more examples](Tutorial/Contributing.md) for other NNI users. 

## Support Library:
NNI also supports all libraries written in python.Here are some common libraries, including some algorithms based on GBDT: XGBoost, CatBoost and lightGBM.
* [Scikit-learn]https://scikit-learn.org/stable/
    <ul>
    <li><a href="TrialExample/SklearnExamples.md">Scikit-learn</a><br/></li>
    </ul>
* [XGBoost]https://xgboost.readthedocs.io/en/latest/
* [CatBoost]https://catboost.ai/
* [LightGBM]https://lightgbm.readthedocs.io/en/latest/
    <ul>
    <li><a href="TrialExample/GbdtExample.md">Auto-gbdt</a><br/></li>
    </ul>
Here is just a small list of libraries that supported by NNI. If you are interested in NNI, you can refer to the [tutorial](TrialExample/Trials.md) to complete your own hacks.



In addition to these experiments, we also welcome more and more users to apply NNI to your own experiment, if you have any doubts, please refer [Write a Trial Run on NNI](TrialExample/Trials.md). In particular, if you want to be a contributor of NNI, whether it is the sharing of examples , writing of Tuner or otherwise, we are all looking forward to your participation.More information please refer to [here](Tutorial/Contributing.md).
