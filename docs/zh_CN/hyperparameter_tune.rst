.. 6ed30d3a87dbc4c1c4650cf56f074045

##############
自动超参数调优
##############

自动调优是 NNI 的主要功能之一。它的工作模式是反复运行 trial 代码，每次向其提供不同的超参组合，从而对 trial 的运行结果进行调优。NNI 提供了很多流行的自动调优算法（称为 Tuner）和一些提前终止算法（称为 Assessor）。NNI 支持在多种训练平台上运行 trial，包括本机、远程服务器、Azure Machine Learning、基于 Kubernetes 的集群（如 OpenPAI、Kubeflow）等等。

NNI 具有高扩展性，用户可以根据需求实现自己的 Tuner 算法和训练平台。

..  toctree::
    :maxdepth: 2

    实现 Trial <./TrialExample/Trials>
    Tuners <builtin_tuner>
    Assessors <builtin_assessor>
    训练平台 <training_services>
    示例 <examples>
    Web 界面 <Tutorial/WebUI>
    如何调试 <Tutorial/HowToDebug>
    高级功能 <hpo_advanced>
    Tuner 基准测试 <hpo_benchmark>
