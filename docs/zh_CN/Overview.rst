概述
========

NNI (Neural Network Intelligence) 是一个工具包，可有效的帮助用户设计并调优机器学习模型的神经网络架构，复杂系统的参数（如超参）等。 NNI 的特性包括：易于使用，可扩展，灵活，高效。


* **易于使用**：NNI 可通过 pip 安装。 只需要在代码中添加几行，就可以利用 NNI 来调优参数。 可使用命令行工具或 Web 界面来查看 Experiment。
* **可扩展**：调优超参或网络结构通常需要大量的计算资源。NNI 在设计时就支持了多种不同的计算资源，如远程服务器组，训练平台（如：OpenPAI，Kubernetes），等等。 根据您配置的培训平台的能力，可以并行运行数百个 Trial 。
* **灵活**：除了内置的算法，NNI 中还可以轻松集成自定义的超参调优算法，神经网络架构搜索算法，提前终止算法等等。 还可以将 NNI 连接到更多的训练平台上，如云中的虚拟机集群，Kubernetes 服务等等。 此外，NNI 还可以连接到外部环境中的特殊应用和模型上。
* **高效**：NNI 在系统及算法级别上不断地进行优化。 例如：通过早期的反馈来加速调优过程。

下图显示了 NNI 的体系结构。


.. raw:: html

   <p align="center">
   <img src="https://user-images.githubusercontent.com/16907603/92089316-94147200-ee00-11ea-9944-bf3c4544257f.png" alt="drawing" width="700"/>
   </p>


主要概念
------------


* 
  *Experiment（实验）*： 表示一次任务，例如，寻找模型的最佳超参组合，或最好的神经网络架构等。 它由 Trial 和自动机器学习算法所组成。

* 
  *搜索空间*：是模型调优的范围。 例如，超参的取值范围。

* 
  *Configuration（配置）*：配置是来自搜索空间的实例，每个超参都会有特定的值。

* 
  *Trial*：是一次独立的尝试，它会使用某组配置（例如，一组超参值，或者特定的神经网络架构）。 Trial 会基于提供的配置来运行。

* 
  *Tuner（调优器）*：一种自动机器学习算法，会为下一个 Trial 生成新的配置。 新的 Trial 会使用这组配置来运行。

* 
  *Assessor（评估器）*：分析 Trial 的中间结果（例如，定期评估数据集上的精度），来确定 Trial 是否应该被提前终止。

* 
  *训练平台*：是 Trial 的执行环境。 根据 Experiment 的配置，可以是本机，远程服务器组，或其它大规模训练平台（如，OpenPAI，Kubernetes）。

Experiment 的运行过程为：Tuner 接收搜索空间并生成配置。 这些配置将被提交到训练平台，如本机，远程服务器组或训练集群。 执行的性能结果会被返回给 Tuner。 然后，再生成并提交新的配置。

每次 Experiment 执行时，用户只需要定义搜索空间，改动几行代码，就能利用 NNI 内置的 Tuner/Assessor 和训练平台来搜索最好的超参组合以及神经网络结构。 基本上分为三步：

..

   步骤一：`定义搜索空间 <Tutorial/SearchSpaceSpec.rst>`__

   步骤二：`改动模型代码 <TrialExample/Trials.rst>`__

   步骤三：`定义实验配置 <Tutorial/ExperimentConfig.rst>`__



.. raw:: html

   <p align="center">
   <img src="https://user-images.githubusercontent.com/23273522/51816627-5d13db80-2302-11e9-8f3e-627e260203d5.jpg" alt="drawing"/>
   </p>


可查看 `快速入门 <Tutorial/QuickStart.rst>`__  来调优你的模型或系统。

核心功能
-------------

NNI 提供了并行运行多个实例以查找最佳参数组合的能力。 此功能可用于各种领域，例如，为深度学习模型查找最佳超参数，或查找具有真实数据的数据库和其他复杂系统的最佳配置。

NNI 还希望提供用于机器学习和深度学习的算法工具包，尤其是神经体系结构搜索（NAS）算法，模型压缩算法和特征工程算法。

超参调优
^^^^^^^^^^^^^^^^^^^^^

这是 NNI 最核心、基本的功能，其中提供了许多流行的 `自动调优算法 <Tuner/BuiltinTuner.rst>`__ （即 Tuner) 以及 `提前终止算法 <Assessor/BuiltinAssessor.rst>`__ （即 Assessor）。 可查看 `快速入门 <Tutorial/QuickStart.rst>`__ 来调优你的模型或系统。 基本上通过以上三步，就能开始 NNI Experiment。

通用 NAS 框架
^^^^^^^^^^^^^^^^^^^^^

此 NAS 框架可供用户轻松指定候选的神经体系结构，例如，可以为单个层指定多个候选操作（例如，可分离的 conv、扩张 conv），并指定可能的跳过连接。 NNI 将自动找到最佳候选。 另一方面，NAS 框架为其他类型的用户（如，NAS 算法研究人员）提供了简单的接口，以实现新的 NAS 算法。 NAS 详情及用法参考 `这里 <NAS/Overview.rst>`__。

NNI 通过 Trial SDK 支持多种 one-shot（一次性） NAS 算法，如：ENAS、DARTS。 使用这些算法时，不需启动 NNI Experiment。 在 Trial 代码中加入算法，直接运行即可。 如果要调整算法中的超参数，或运行多个实例，可以使用 Tuner 并启动 NNI Experiment。

除了 one-shot NAS 外，NAS 还能以 NNI 模式运行，其中每个候选的网络结构都作为独立 Trial 任务运行。 在此模式下，与超参调优类似，必须启动 NNI Experiment 并为 NAS 选择 Tuner。

模型压缩
^^^^^^^^^^^^^^^^^

NNI 提供了一个易于使用的模型压缩框架来压缩深度神经网络，压缩后的网络通常具有更小的模型尺寸和更快的推理速度，
模型性能也不会有明显的下降。 NNI 上的模型压缩包括剪枝和量化算法。 这些算法通过 NNI Trial SDK 提供
。 可以直接在 Trial 代码中使用，并在不启动 NNI Experiment 的情况下运行 Trial 代码。 用户还可以使用 NNI 模型压缩框架集成自定义的剪枝和量化算法。

模型压缩的详细说明和算法可在 `这里 <Compression/Overview.rst>`__ 找到。

自动特征工程
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

自动特征工程，可以为下游任务找到最有效的特征。 自动特征工程及其用法的详细说明可在 `这里 <FeatureEngineering/Overview.rst>`__ 找到。 通过 NNI Trial SDK 支持，不必创建 NNI Experiment， 只需在 Trial 代码中加入内置的自动特征工程算法，然后直接运行 Trial 代码。 

自动特征工程算法通常有一些超参。 如果要自动调整这些超参，可以利用 NNI 的超参数调优，即选择调优算法（即 Tuner）并启动 NNI Experiment。

了解更多信息
--------------------


* `入门 <Tutorial/QuickStart.rst>`__
* `如何为 NNI 调整代码？ <TrialExample/Trials.rst>`__
* `NNI 支持哪些 Tuner？ <Tuner/BuiltinTuner.rst>`__
* `如何自定义 Tuner？ <Tuner/CustomizeTuner.rst>`__
* `NNI 支持哪些 Assessor？ <Assessor/BuiltinAssessor.rst>`__
* `如何自定义 Assessor？ <Assessor/CustomizeAssessor.rst>`__
* `如何在本机上运行 Experiment？ <TrainingService/LocalMode.rst>`__
* `如何在多机上运行 Experiment？ <TrainingService/RemoteMachineMode.rst>`__
* `如何在 OpenPAI 上运行 Experiment？ <TrainingService/PaiMode.rst>`__
* `示例 <TrialExample/MnistExamples.rst>`__
* `NNI 上的神经网络架构搜索 <NAS/Overview.rst>`__
* `NNI 上的自动模型压缩 <Compression/Overview.rst>`__
* `NNI 上的自动特征工程 <FeatureEngineering/Overview.rst>`__
