.. dbd41cab307bcd76cc747b3d478709b8


NNI 文档
=================

.. toctree::
   :maxdepth: 2
   :caption: 开始使用
   :hidden:

   安装 <installation>
   快速入门 <quickstart>

.. toctree::
   :maxdepth: 2
   :caption: 用户指南
   :hidden:

   超参调优 <hpo/toctree>
   架构搜索 <nas/toctree>
   模型压缩 <compression/toctree>
   特征工程 <feature_engineering/toctree>
   实验管理 <experiment/toctree>

.. toctree::
   :maxdepth: 2
   :caption: 参考
   :hidden:

   Python API <reference/python_api>
   实验配置 <reference/experiment_config>
   nnictl 命令 <reference/nnictl>

.. toctree::
   :maxdepth: 2
   :caption: 杂项
   :hidden:

   示例 <examples>
   社区分享 <sharings/community_sharings>
   研究发布 <notes/research_publications>
   源码安装 <notes/build_from_source>
   贡献指南 <notes/contributing>
   版本说明 <release>

**NNI (Neural Network Intelligence)** 是一个轻量而强大的工具，可以帮助用户 **自动化**：

* :doc:`超参调优 </hpo/overview>`
* :doc:`架构搜索 </nas/overview>`
* :doc:`模型压缩 </compression/overview>`
* :doc:`特征工程 </feature_engineering/overview>`

开始使用
-----------

安装最新的版本，可执行以下命令：

.. code-block:: bash

   $ pip install nni

如果在安装上遇到问题，可参考 :doc:`安装指南 </installation>`。

开始你的第一个 NNI 实验
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   $ nnictl hello

.. note:: 你需要预先安装 `PyTorch <https://pytorch.org/>`_ （以及 `torchvision <https://pytorch.org/vision/stable/index.html>`_ ）才能运行这个实验。

请阅读 :doc:`NNI 快速入门 <quickstart>` 以开启你的 NNI 旅程！

为什么选择 NNI？
--------------------

NNI 使得自动机器学习技术即插即用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <div class="codesnippet-card-container">

.. codesnippetcard::
   :icon: ../img/thumbnails/hpo-small.svg
   :title: 超参调优
   :link: tutorials/hpo_quickstart_pytorch/main
   :seemore: 点这里阅读完整教程

   .. code-block::

      params = nni.get_next_parameter()

      class Net(nn.Module):
          ...

      model = Net()
      optimizer = optim.SGD(model.parameters(),
                            params['lr'],
                            params['momentum'])

      for epoch in range(10):
          train(...)

      accuracy = test(model)
      nni.report_final_result(accuracy)

.. codesnippetcard::
   :icon: ../img/thumbnails/pruning-small.svg
   :title: 模型剪枝
   :link: tutorials/pruning_quick_start
   :seemore: 点这里阅读完整教程

   .. code-block::

      # define a config_list
      config = [{
          'sparsity': 0.8,
          'op_types': ['Conv2d']
      }]

      # generate masks for simulated pruning
      wrapped_model, masks = \
          L1NormPruner(model, config). \
          compress()

      # apply the masks for real speedup
      ModelSpeedup(unwrapped_model, input, masks). \
          speedup_model()

.. codesnippetcard::
   :icon: ../img/thumbnails/quantization-small.svg
   :title: 模型量化
   :link: tutorials/quantization_speedup
   :seemore: 点这里阅读完整教程

   .. code-block::

      # define a config_list
      config = [{
          'quant_types': ['input', 'weight'],
          'quant_bits': {'input': 8, 'weight': 8},
          'op_types': ['Conv2d']
      }]

      # in case quantizer needs a extra training
      quantizer = QAT_Quantizer(model, config)
      quantizer.compress()
      # Training...

      # export calibration config and
      # generate TensorRT engine for real speedup
      calibration_config = quantizer.export_model(
          model_path, calibration_path)
      engine = ModelSpeedupTensorRT(
          model, input_shape, config=calib_config)
      engine.compress()

.. codesnippetcard::
   :icon: ../img/thumbnails/multi-trial-nas-small.svg
   :title: 神经网络架构搜索
   :link: tutorials/hello_nas
   :seemore: 点这里阅读完整教程

   .. code-block::

      # define model space
      -   self.conv2 = nn.Conv2d(32, 64, 3, 1)
      +   self.conv2 = nn.LayerChoice([
      +       nn.Conv2d(32, 64, 3, 1),
      +       DepthwiseSeparableConv(32, 64)
      +   ])
      # search strategy + evaluator
      strategy = RegularizedEvolution()
      evaluator = FunctionalEvaluator(
          train_eval_fn)

      # run experiment
      RetiariiExperiment(model_space,
          evaluator, strategy).run()

.. codesnippetcard::
   :icon: ../img/thumbnails/one-shot-nas-small.svg
   :title: 单尝试 (One-shot) NAS
   :link: nas/exploration_strategy
   :seemore: 点这里阅读完整教程

   .. code-block::

      # define model space
      space = AnySearchSpace()

      # get a darts trainer
      trainer = DartsTrainer(space, loss, metrics)
      trainer.fit()

      # get final searched architecture
      arch = trainer.export()

.. codesnippetcard::
   :icon: ../img/thumbnails/feature-engineering-small.svg
   :title: 特征工程
   :link: feature_engineering/overview
   :seemore: 点这里阅读完整教程

   .. code-block::

      selector = GBDTSelector()
      selector.fit(
          X_train, y_train,
          lgb_params=lgb_params,
          eval_ratio=eval_ratio,
          early_stopping_rounds=10,
          importance_type='gain',
          num_boost_round=1000)

      # get selected features
      features = selector.get_selected_features()

.. End of code snippet card

.. raw:: html

   </div>

NNI 可降低自动机器学习实验管理的成本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. codesnippetcard::
   :icon: ../img/thumbnails/training-service-small.svg
   :title: 训练平台
   :link: experiment/training_service/overview
   :seemore: 点这里了解更多

   一个自动机器学习实验通常需要很多次尝试，来找到合适且具有潜力的模型。
   **训练平台** 的目标便是让整个调优过程可以轻松的扩展到分布式平台上，为不同的计算资源（例如本地机器、远端服务器、集群等）提供的统一的用户体验。
   目前，NNI 已经支持 **超过九种** 训练平台。

.. codesnippetcard::
   :icon: ../img/thumbnails/web-portal-small.svg
   :title: 网页控制台
   :link: experiment/web_portal/web_portal
   :seemore: 点这里了解更多

   网页控制台提供了可视化调优过程的能力，让你可以轻松检查、跟踪、控制实验流程。

   .. image:: ../static/img/webui.gif
      :width: 100%

.. codesnippetcard::
   :icon: ../img/thumbnails/experiment-management-small.svg
   :title: 多实验管理
   :link: experiment/experiment_management
   :seemore: 点这里了解更多

   深度学习模型往往需要多个实验不断迭代，例如用户可能想尝试不同的调优算法，优化他们的搜索空间，或者切换到其他的计算资源。
   **多实验管理** 提供了对多个实验的结果进行聚合和比较的强大能力，极大程度上简化了开发者的开发流程。

获取帮助或参与贡献
-------------------------------

NNI 使用 `NNI GitHub 仓库 <https://github.com/microsoft/nni>`_ 进行维护。我们在 GitHub 上收集反馈，以及新需求和想法。你可以：

* 新建一个 `GitHub issue <https://github.com/microsoft/nni/issues>`_ 反馈一个 bug 或者需求。
* 新建一个 `pull request <https://github.com/microsoft/nni/pulls>`_ 以贡献代码（在此之前，请务必确保你已经阅读过 :doc:`贡献指南 <notes/contributing>`）。
* 如果你有任何问题，都可以加入 `NNI 讨论 <https://github.com/microsoft/nni/discussions>`_。
* 加入即时聊天群组：

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Gitter
     - 微信
   * -
       .. image:: https://user-images.githubusercontent.com/39592018/80665738-e0574a80-8acc-11ea-91bc-0836dc4cbf89.png
     -
       .. image:: https://github.com/scarlett2018/nniutil/raw/master/wechat.png

引用 NNI
----------

如果你在你的文献中用到了 NNI，请考虑引用我们：

   Microsoft. Neural Network Intelligence (version |release|). https://github.com/microsoft/nni

Bibtex 格式如下（请将版本号替换成你在使用的特定版本）： ::

   @software{nni2021,
      author = {{Microsoft}},
      month = {1},
      title = {{Neural Network Intelligence}},
      url = {https://github.com/microsoft/nni},
      version = {2.0},
      year = {2021}
   }
