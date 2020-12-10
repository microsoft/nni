.. role:: raw-html(raw)
   :format: html


Framework and Library Supports
==============================

With the built-in Python API, NNI naturally supports the hyper parameter tuning and neural network search for all the AI frameworks and libraries who support Python models(\ ``version >= 3.6``\ ). NNI had also provided a set of examples and tutorials for some of the popular scenarios to make jump start easier.

Supported AI Frameworks
-----------------------


* :raw-html:`<b>[PyTorch]</b>` https://github.com/pytorch/pytorch

  .. raw:: html

     <ul> 
         <li><a href="../../examples/trials/mnist-distributed-pytorch">MNIST-pytorch</a><br/></li>
         <li><a href="TrialExample/Cifar10Examples.md">CIFAR-10</a><br/></li>
         <li><a href="../../examples/trials/kaggle-tgs-salt/README.md">TGS salt identification chanllenge</a><br/></li>
         <li><a href="../../examples/trials/network_morphism/README.md">Network_morphism</a><br/></li>
       </ul>


* :raw-html:`<b>[TensorFlow]</b>` https://github.com/tensorflow/tensorflow

  .. raw:: html

     <ul> 
         <li><a href="../../examples/trials/mnist-distributed">MNIST-tensorflow</a><br/></li>
          <li><a href="../../examples/trials/ga_squad/README.md">Squad</a><br/></li>
       </ul>


* :raw-html:`<b>[Keras]</b>` https://github.com/keras-team/keras

  .. raw:: html

     <ul>
         <li><a href="../../examples/trials/mnist-keras">MNIST-keras</a><br/></li>
         <li><a href="../../examples/trials/network_morphism/README.md">Network_morphism</a><br/></li>
       </ul>


* :raw-html:`<b>[MXNet]</b>` https://github.com/apache/incubator-mxnet
* :raw-html:`<b>[Caffe2]</b>` https://github.com/BVLC/caffe
* :raw-html:`<b>[CNTK (Python language)]</b>` https://github.com/microsoft/CNTK
* :raw-html:`<b>[Spark MLlib]</b>` http://spark.apache.org/mllib/
* :raw-html:`<b>[Chainer]</b>` https://chainer.org/
* :raw-html:`<b>[Theano]</b>` https://pypi.org/project/Theano/ :raw-html:`<br/>`

You are encouraged to `contribute more examples <Tutorial/Contributing.rst>`__ for other NNI users. 

Supported Library
-----------------

NNI also supports all libraries written in python.Here are some common libraries, including some algorithms based on GBDT: XGBoost, CatBoost and lightGBM.


* :raw-html:`<b>[Scikit-learn]</b>` https://scikit-learn.org/stable/

  .. raw:: html

     <ul>
       <li><a href="TrialExample/SklearnExamples.md">Scikit-learn</a><br/></li>
       </ul>


* :raw-html:`<b>[XGBoost]</b>` https://xgboost.readthedocs.io/en/latest/
* :raw-html:`<b>[CatBoost]</b>` https://catboost.ai/
* :raw-html:`<b>[LightGBM]</b>` https://lightgbm.readthedocs.io/en/latest/
    :raw-html:`<ul>
    <li><a href="TrialExample/GbdtExample.md">Auto-gbdt</a><br/></li>
    </ul>`

Here is just a small list of libraries that supported by NNI. If you are interested in NNI, you can refer to the `tutorial <TrialExample/Trials.rst>`__ to complete your own hacks.

In addition to the above examples, we also welcome more and more users to apply NNI to your own work, if you have any doubts, please refer `Write a Trial Run on NNI <TrialExample/Trials.md>`__. In particular, if you want to be a contributor of NNI, whether it is the sharing of examples , writing of Tuner or otherwise, we are all looking forward to your participation.More information please refer to `here <Tutorial/Contributing.rst>`__.
