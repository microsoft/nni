.. de9c05c0da9751f920113d0b169494a2

快速入门
==========

安装
----

目前NNI支持了 Linux、macOS 和 Windows系统。 其中，Ubuntu 16.04 及更高版本、macOS 10.14.1 和 Windows 10.1809 均经过测试并支持。 在 ``python >= 3.6`` 环境中，只需运行 ``pip install`` 即可完成安装。

Linux 和 macOS
^^^^^^^^^^^^^^

.. code-block:: bash

   python3 -m pip install --upgrade nni

Windows
^^^^^^^

.. code-block:: bash

   python -m pip install --upgrade nni

.. Note:: 在 Linux 和 macOS 上，如果要将 NNI 安装到当前用户的 home 目录中，可使用 ``--user`` ；这不需要特殊权限。

.. Note:: 如果出现 ``Segmentation fault`` 这样的错误，参考 :doc:`常见问题 <FAQ>` 。

.. Note:: NNI 的系统需求，参考 :doc:`Linux & Mac <InstallationLinux>` 或者 :doc:`Windows <InstallationWin>` 的安装教程。如果想要使用 docker, 参考 :doc:`如何使用 Docker <HowToUseDocker>` 。


MNIST 上的 "Hello World"
------------------------

NNI 是一个能进行自动机器学习实验的工具包。 它可以自动进行获取超参、运行 Trial，测试结果，调优超参的循环。 在这里，将演示如何使用 NNI 帮助找到 MNIST 模型的最佳超参数。

这是还 **没有 NNI** 的示例代码，用 CNN 在 MNIST 数据集上训练：

.. code-block:: python

    def main(args):
        # 下载数据
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(...), batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.tuils.data.DataLoader(datasets.MNIST(...), batch_size=1000, shuffle=True)
        # 构建模型
        model = Net(hidden_size=args['hidden_size'])
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
        # 训练
        for epoch in range(10):
            train(args, model, device, train_loader, optimizer, epoch)
            test_acc = test(args, model, device, test_loader)
            print(test_acc)
        print('final accuracy:', test_acc)
         
    if __name__ == '__main__':
        params = {
            'batch_size': 32,
            'hidden_size': 128,
            'lr': 0.001,
            'momentum': 0.5
        }
        main(params)

上面的代码一次只能尝试一组参数，如果想要调优学习率，需要手工改动超参，并一次次尝试。

NNI 用来帮助超参调优。它的流程如下：

.. code-block:: text

   输入: 搜索空间, Trial 代码, 配置文件
   输出: 一组最优的参数配置

   1: For t = 0, 1, 2, ..., maxTrialNum,
   2:      hyperparameter = 从搜索空间选择一组参数
   3:      final result = run_trial_and_evaluate(hyperparameter)
   4:      返回最终结果给 NNI
   5:      If 时间达到上限,
   6:          停止实验
   7: 返回最好的实验结果

.. note::

   如果需要使用 NNI 来自动训练模型，找到最佳超参，有两种实现方式：

   1. 编写配置文件，然后使用命令行启动 experiment；
   2. 直接从 Python 文件中配置并启动 experiment。

   在本节中，我们将重点介绍第一种实现方式。如果希望使用第二种实现方式，请参考 `教程 <HowToLaunchFromPython.rst>`__\ 。


第一步：修改 ``Trial`` 代码
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

修改 ``Trial`` 代码来从 NNI 获取超参，并向 NNI 报告训练结果。

.. code-block:: diff

    + import nni

      def main(args):
          # 下载数据
          train_loader = torch.utils.data.DataLoader(datasets.MNIST(...), batch_size=args['batch_size'], shuffle=True)
          test_loader = torch.tuils.data.DataLoader(datasets.MNIST(...), batch_size=1000, shuffle=True)
          # 构造模型
          model = Net(hidden_size=args['hidden_size'])
          optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
          # 训练
          for epoch in range(10):
              train(args, model, device, train_loader, optimizer, epoch)
              test_acc = test(args, model, device, test_loader)
    -         print(test_acc)
    +         nni.report_intermeidate_result(test_acc)
    -     print('final accuracy:', test_acc)
    +     nni.report_final_result(test_acc)
           
      if __name__ == '__main__':
    -     params = {'batch_size': 32, 'hidden_size': 128, 'lr': 0.001, 'momentum': 0.5}
    +     params = nni.get_next_parameter()
          main(params)

*示例：* :githublink:`mnist.py <examples/trials/mnist-pytorch/mnist.py>`


第二步：定义搜索空间
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

编写 YAML 格式的 **搜索空间** 文件，包括所有需要搜索的超参的 **名称** 和 **分布** （离散和连续值均可）。

.. code-block:: yaml

   searchSpace:
      batch_size:
         _type: choice
         _value: [16, 32, 64, 128]
      hidden_size:
         _type: choice
         _value: [128, 256, 512, 1024]
      lr:
         _type: choice
         _value: [0.0001, 0.001, 0.01, 0.1]
      momentum:
         _type: uniform
         _value: [0, 1]

*示例：* :githublink:`config_detailed.yml <examples/trials/mnist-pytorch/config_detailed.yml>`

也可以使用 JSON 文件来编写搜索空间，并在配置中确认文件路径。关于如何编写搜索空间，可以参考 `教程 <SearchSpaceSpec.rst>`__.


第三步：配置 experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

除了在第二步中定义的搜索空间，还需要定义 YAML 格式的 **配置** 文件，声明 experiment 的关键信息，例如 Trail 文件，调优算法，最大 Trial 运行次数和最大持续时间等。

.. code-block:: yaml

   experimentName: MNIST               # 用于区分 experiment 的名字，可选项
   trialCommand: python3 mnist.py      # 注意：如果使用 Windows，请将 "python3" 修改为 "python" 
   trialConcurrency: 2                 # 同时运行 2 个 trial
   maxTrialNumber: 10                  # 最多生成 10 个 trial
   maxExperimentDuration: 1h           # 1 小时后停止生成 trial
   tuner:                              # 配置调优算法
      name: TPE
      classArgs:                       # 算法特定参数
         optimize_mode: maximize
   trainingService:                    # 配置训练平台
      platform: local

Experiment 的配置文件可以参考 `文档 <../reference/experiment_config.rst>`__.

.. _nniignore:

.. Note:: 如果要使用远程服务器或集群作为 :doc:`训练平台 <../TrainingService/Overview>`，为了避免产生过大的网络压力，NNI 限制了文件的最大数量为 2000，大小为 300 MB。 如果代码目录中包含了过多的文件，可添加 ``.nniignore`` 文件来排除部分，与 ``.gitignore`` 文件用法类似。 参考 `git documentation <https://git-scm.com/docs/gitignore#_pattern_format>`__ ，了解更多如何编写此文件的详细信息。

*示例：* :githublink:`config.yml <examples/trials/mnist-pytorch/config.yml>` 和 :githublink:`.nniignore <examples/trials/mnist-pytorch/.nniignore>`

上面的代码都已准备好，并保存在 :githublink:`examples/trials/mnist-pytorch/ <examples/trials/mnist-pytorch>`。


第四步：运行 experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linux 和 macOS
**************

从命令行使用 **config.yml** 文件启动 MNIST experiment 。

.. code-block:: bash

   nnictl create --config nni/examples/trials/mnist-pytorch/config_detailed.yml

Windows
*******

在 **config_detailed.yml** 文件的 ``trialCommand`` 项中将 ``python3`` 修改为 ``python``，然后从命令行使用 **config_detailed.yml** 文件启动 MNIST experiment 。

.. code-block:: bash

   nnictl create --config nni\examples\trials\mnist-pytorch\config_detailed.yml

.. Note:: ``nnictl`` 是一个命令行工具，用来控制 NNI experiment，如启动、停止、继续 experiment，启动、停止 NNIBoard 等等。 点击 :doc:`这里 <Nnictl>` 查看 ``nnictl`` 的更多用法。

在命令行中等待输出 ``INFO: Successfully started experiment!`` 。 此消息表明实验已成功启动。 期望的输出如下：

.. code-block:: text

   INFO: Starting restful server...
   INFO: Successfully started Restful server!
   INFO: Setting local config...
   INFO: Successfully set local config!
   INFO: Starting experiment...
   INFO: Successfully started experiment!
   -----------------------------------------------------------------------
   The experiment id is egchD4qy
   The Web UI urls are: [Your IP]:8080
   -----------------------------------------------------------------------

   You can use these commands to get more information about the experiment
   -----------------------------------------------------------------------
            commands                       description
   1. nnictl experiment show        show the information of experiments
   2. nnictl trial ls               list all of trial jobs
   3. nnictl top                    monitor the status of running experiments
   4. nnictl log stderr             show stderr log content
   5. nnictl log stdout             show stdout log content
   6. nnictl stop                   stop an experiment
   7. nnictl trial kill             kill a trial job by id
   8. nnictl --help                 get help information about nnictl
   -----------------------------------------------------------------------

如果根据上述步骤准备好了相应 ``Trial`` ， **搜索空间** 和 **配置** ，并成功创建的 NNI 任务。NNI 会自动开始通过配置的搜索空间来运行不同的超参集合，搜索最好的超参。 通过 Web 界面可看到 NNI 的进度。

第五步：查看 experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

启动 experiment 后，可以在命令行界面找到如下的 **Web 界面地址** ：

.. code-block:: text

   The Web UI urls are: [Your IP]:8080

在浏览器中打开 **Web 界面地址** （即： ``[IP 地址]:8080`` ），就可以看到 experiment 的详细信息，以及所有的 Trial 任务。 如果无法打开终端中的 Web 界面链接，可以参考 `常见问题 <FAQ.rst>`__。


查看概要页面
******************

Experiment 相关信息会显示在界面上，包括配置和搜索空间等。 NNI 还支持通过 **Experiment summary** 按钮下载这些信息和参数。

.. image:: ../../img/webui-img/full-oview.png
   :target: ../../img/webui-img/full-oview.png
   :alt: overview


查看 Trial 详情页面
**********************************

可以在此页面中看到最佳的 ``Trial`` 指标和超参数图。 您可以点击 ``Add/Remove columns`` 按钮向表格中添加更多列。

.. image:: ../../img/webui-img/full-detail.png
   :target: ../../img/webui-img/full-detail.png
   :alt: detail


查看 experiment 管理页面
**********************************

``All experiments`` 页面可以查看计算机上的所有实验。 

.. image:: ../../img/webui-img/managerExperimentList/expList.png
   :target: ../../img/webui-img/managerExperimentList/expList.png
   :alt: Experiments list

更多信息可参考 `此文档 <./WebUI.rst>`__。


相关主题
-------------

* `进行Debug <HowToDebug.rst>`__
* `如何实现 Trial 代码 <../TrialExample/Trials.rst>`__
* `尝试不同的 Tuner <../Tuner/BuiltinTuner.rst>`__
* `尝试不同的 Assessor <../Assessor/BuiltinAssessor.rst>`__
* `在不同训练平台上运行 experiment <../training_services.rst>`__
* `如何使用 Annotation <AnnotationSpec.rst>`__
* `如何使用命令行工具 nnictl <Nnictl.rst>`__
* `在 Web 界面中启动 TensorBoard <Tensorboard.rst>`__
