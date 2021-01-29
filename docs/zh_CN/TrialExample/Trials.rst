.. role:: raw-html(raw)
   :format: html


实现 NNI 的 Trial（尝试）代码
===========================================

**Trial（尝试）** 是将一组参数组合（例如，超参）在模型上独立的一次尝试。

定义 NNI 的 Trial，需要首先定义参数组（例如，搜索空间），并更新模型代码。 有两种方法来定义一个 Trial：`NNI API <#nni-api>`__ 和 `NNI Python annotation <#nni-annotation>`__。 参考 `这里 <#more-examples>`__ 更多 Trial 示例。

:raw-html:`<a name="nni-api"></a>`

NNI API
-------

第一步：准备搜索空间参数文件。
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

样例如下：

.. code-block:: json

   {
       "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
       "conv_size":{"_type":"choice","_value":[2,3,5,7]},
       "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
       "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
   }

参考 `SearchSpaceSpec.md <../Tutorial/SearchSpaceSpec.rst>`__ 进一步了解搜索空间。 Tuner 会根据搜索空间来生成配置，即从每个超参的范围中选一个值。

第二步：更新模型代码
^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  Import NNI

    在 Trial 代码中加上 ``import nni`` 。

* 
  从 Tuner 获得参数值

.. code-block:: python

   RECEIVED_PARAMS = nni.get_next_parameter()

``RECEIVED_PARAMS`` 是一个对象，如：

``{"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}``


* 定期返回指标数据（可选）

.. code-block:: python

   nni.report_intermediate_result(metrics)

``指标`` 可以是任意的 Python 对象。 如果使用了 NNI 内置的 Tuner/Assessor，``指标`` 只可以是两种类型：1) 数值类型，如 float、int， 2) dict 对象，其中必须由键名为 ``default`` ，值为数值的项目。 ``指标`` 会发送给 `assessor <../Assessor/BuiltinAssessor.rst>`__。 通常，``指标`` 包含了定期评估的损失值或精度。


* 返回配置的最终性能

.. code-block:: python

   nni.report_final_result(metrics)

``指标`` 可以是任意的 Python 对象。 如果使用了内置的 Tuner/Assessor，``指标`` 格式和 ``report_intermediate_result`` 中一样，这个数值表示模型的性能，如精度、损失值等。 ``指标`` 会发送给 `tuner <../Tuner/BuiltinTuner.rst>`__。

第三步：启用 NNI API
^^^^^^^^^^^^^^^^^^^^^^^

要启用 NNI 的 API 模式，需要将 useAnnotation 设置为 *false*，并提供搜索空间文件的路径，即第一步中定义的文件：

.. code-block:: yaml

   useAnnotation: false
   searchSpacePath: /path/to/your/search_space.json

参考 `这里 <../Tutorial/ExperimentConfig.rst>`__ 进一步了解如何配置 Experiment。

参考 `这里 </sdk_reference.html>`__ ，了解更多 NNI API （例如：``nni.get_sequence_id()``）。

:raw-html:`<a name="nni-annotation"></a>`

NNI Annotation
---------------------

另一种实现 Trial 的方法是使用 Python 注释来标记 NNI。 NN Annotation 很简单，类似于注释。 不必对现有代码进行结构更改。 只需要添加一些 NNI Annotation，就能够：


* 标记需要调整的参数变量
* 指定要在其中调整的变量的范围
* 标记哪个变量需要作为中间结果范围给 ``assessor``
* 标记哪个变量需要作为最终结果（例如：模型精度） 返回给 ``tuner``

同样以 MNIST 为例，只需要两步就能用 NNI Annotation 来实现 Trial 代码。

第一步：在代码中加入 Annotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

下面是加入了 Annotation 的 TensorFlow 代码片段，高亮的 4 行 Annotation 用于：


#. 调优 batch_size 和 dropout_rate
#. 每执行 100 步返回 test_acc
#. 最后返回 test_acc 作为最终结果。

值得注意的是，新添加的代码都是注释，不会影响以前的执行逻辑。因此这些代码仍然能在没有安装 NNI 的环境中运行。

.. code-block:: diff

   with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
   +   """@nni.variable(nni.choice(50, 250, 500), name=batch_size)"""
       batch_size = 128
       for i in range(10000):
           batch = mnist.train.next_batch(batch_size)
   +       """@nni.variable(nni.choice(0.1, 0.5), name=dropout_rate)"""
           dropout_rate = 0.5
           mnist_network.train_step.run(feed_dict={mnist_network.images: batch[0],
                                                   mnist_network.labels: batch[1],
                                                   mnist_network.keep_prob: dropout_rate})
           if i % 100 == 0:
               test_acc = mnist_network.accuracy.eval(
                   feed_dict={mnist_network.images: mnist.test.images,
                               mnist_network.labels: mnist.test.labels,
                               mnist_network.keep_prob: 1.0})
   +           """@nni.report_intermediate_result(test_acc)"""

       test_acc = mnist_network.accuracy.eval(
           feed_dict={mnist_network.images: mnist.test.images,
                       mnist_network.labels: mnist.test.labels,
                       mnist_network.keep_prob: 1.0})
   +   """@nni.report_final_result(test_acc)"""

**注意**：


* ``@nni.variable`` 会对它的下面一行进行修改，左边被赋值变量必须与 ``@nni.variable`` 的关键字 ``name`` 相同。
* ``@nni.report_intermediate_result``\ /\ ``@nni.report_final_result`` 会将数据发送给 assessor/tuner。

Annotation 的语法和用法等，参考 `Annotation <../Tutorial/AnnotationSpec.rst>`__。

第二步：启用 Annotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 YAML 配置文件中设置 *useAnnotation* 为 true 来启用 Annotation：

.. code-block:: bash

   useAnnotation: true

用于调试的独立模式
-----------------------------

NNI 支持独立模式，使 Trial 代码无需启动 NNI 实验即可运行。 这样能更容易的找出 Trial 代码中的 Bug。 NNI Annotation 天然支持独立模式，因为添加的 NNI 相关的行都是注释的形式。 NNI Trial API 在独立模式下的行为有所变化，某些 API 返回虚拟值，而某些 API 不报告值。 有关这些 API 的完整列表，请参阅下表。

.. code-block:: python

   # 注意：请为 Trial 代码中的超参分配默认值
   nni.report_final_result # 已在 stdout 上打印日志，但不报告
   nni.report_intermediate_result # 已在 stdout 上打印日志，但不报告
   nni.get_experiment_id # 返回 "STANDALONE"
   nni.get_trial_id # 返回 "STANDALONE"
   nni.get_sequence_id # 返回 0

可使用 :githublink:`mnist 示例 <examples/trials/mnist-tfv1>` 来尝试独立模式。 只需在代码目录下运行 ``python3 mnist.py``。 Trial 代码会使用默认超参成功运行。

更多调试的信息，可参考 `How to Debug <../Tutorial/HowToDebug.rst>`__。

Trial 存放在什么地方？
----------------------------------------

本机模式
^^^^^^^^^^

每个 Trial 都有单独的目录来输出自己的数据。 在每次 Trial 运行后，环境变量 ``NNI_OUTPUT_DIR`` 定义的目录都会被导出。 在这个目录中可以看到 Trial 的代码、数据和日志。 此外，Trial 的日志（包括 stdout）还会被重定向到此目录中的 ``trial.log`` 文件。

如果使用了 Annotation 方法，转换后的 Trial 代码会存放在另一个临时目录中。 可以在 ``run.sh`` 文件中的 ``NNI_OUTPUT_DIR`` 变量找到此目录。 文件中的第二行（即：``cd``）会切换到代码所在的实际路径。 ``run.sh`` 文件示例：

.. code-block:: bash

   #!/bin/bash
   cd /tmp/user_name/nni/annotation/tmpzj0h72x6 #This is the actual directory
   export NNI_PLATFORM=local
   export NNI_SYS_DIR=/home/user_name/nni-experiments/$experiment_id$/trials/$trial_id$
   export NNI_TRIAL_JOB_ID=nrbb2
   export NNI_OUTPUT_DIR=/home/user_name/nni-experiments/$eperiment_id$/trials/$trial_id$
   export NNI_TRIAL_SEQ_ID=1
   export MULTI_PHASE=false
   export CUDA_VISIBLE_DEVICES=
   eval python3 mnist.py 2>/home/user_name/nni-experiments/$experiment_id$/trials/$trial_id$/stderr
   echo $? `date +%s%3N` >/home/user_name/nni-experiments/$experiment_id$/trials/$trial_id$/.nni/state

其它模式
^^^^^^^^^^^

当 Trial 运行在 OpenPAI 这样的远程服务器上时，``NNI_OUTPUT_DIR`` 仅会指向 Trial 的输出目录，而 ``run.sh`` 不会在此目录中。 ``trial.log`` 文件会被复制回本机的 Trial 目录中。目录的默认位置在 ``~/nni-experiments/$experiment_id$/trials/$trial_id$/``。

更多调试的信息，可参考 `How to Debug <../Tutorial/HowToDebug.rst>`__。

:raw-html:`<a name="more-examples"></a>`

更多 Trial 的示例
-------------------


* `MNIST 示例 <MnistExamples.rst>`__
* `为 CIFAR 10 分类找到最佳的 optimizer <Cifar10Examples.rst>`__
* `如何在 NNI 调优 SciKit-learn 的参数 <SklearnExamples.rst>`__
* `在阅读理解上使用自动模型架构搜索。 <SquadEvolutionExamples.rst>`__
* `如何在 NNI 上调优 GBDT <GbdtExample.rst>`__
* `在 NNI 上调优 RocksDB <RocksdbExamples.rst>`__
