.. role:: raw-html(raw)
   :format: html


框架和库的支持
==============================

通过内置的 Python API，NNI 天然支持所有 Python（ ``版本 >= 3.6`` ）语言的 AI 框架，可使用所有超参调优和神经网络搜索算法。 NNI 还为常见场景提供了一些示例和教程，使上手更容易。

支持的 AI 框架
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

鼓励您 `贡献更多示例 <Tutorial/Contributing.rst>`__ 为其他的 NNI 用户 

支持的库
-----------------

NNI 也支持其它 Python 库，包括一些基于 GBDT 的算法：XGBoost, CatBoost 以及 lightGBM。


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

这只是 NNI 支持的一小部分库。 如果对 NNI 感兴趣，可参考 `此教程 <TrialExample/Trials.rst>`__ 来继续学习。

除了这些案例，也欢迎更多的用户将 NNI 应用到自己的工作中，如果有任何疑问，请参考 `在 NNI 上实现 Trial <TrialExample/Trials.md>`__ 。 如果想成为 NNI 的贡献者，无论是分享示例，还是实现 Tuner 或其它内容，我们都非常期待您的参与。更多信息请参考 `这里 <Tutorial/Contributing.rst>`__ 。
