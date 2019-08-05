# Framework and Library Supported 
NNI provides Python API, supporting all framework models and libraries written in python (`version >= 3.5`), and we have implemented many related examples and detailed their principles and running steps.



## Support Framework:
NNI supports all frameworks, as long as they are written in PYTHON. Some common frameworks are listed below, including NNI case tutorials based on them.

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
* [Scikit-learn]https://scikit-learn.org/stable/
    <ul>
    <li><a href="TrialExample/SklearnExamples.md">Scikit-learn</a><br/></li>
    </ul>
* [Spark MLlib]http://spark.apache.org/mllib/
* [Chainer]https://chainer.org/
* [Theano]https://pypi.org/project/Theano/

## Support Library:
NNI also supports all libraries written in python.Here are some algorithms based on the GBDT library and an example of LIGHTGBM algorithms.

* [XGBoost]https://xgboost.readthedocs.io/en/latest/
* [CatBoost]https://catboost.ai/
* [LightGBM]https://lightgbm.readthedocs.io/en/latest/
    <ul>
    <li><a href="TrialExample/GbdtExample.md">Auto-gbdt</a><br/></li>
    </ul>


If you want to learn how to write a trial and run it on NNI, you can refer to the [Tutorial](TrialExample/Trials.md) for more help. <br/>
In addition to these experiments, we also welcome more and more users to apply NNI to your own experiment. In particular, if you want to be a contributor of NNI, whether it is the sharing of examples , writing of Tuner or otherwise, we are all looking forward to your participation.More information please refer to [here](Tutorial/Contributing.md).
