#############################
自动（超参数）调优
#############################

自动调优是 NNI 提供的关键功能之一，主要应用场景是
超参调优。 应用于 Trial 代码的调优。 提供了很多流行的
自动调优算法（称为 Tuner ）和一些提前终止算法（称为 Assessor）。
NNI 支持在各种培训平台上运行 Trial，例如，在本地计算机上运行，
在多台服务器上分布式运行，或在 OpenPAI，Kubernetes 等平台上。

NNI 的其它重要功能，例如模型压缩，特征工程，也可以进一步
通过自动调优来提高，这会在介绍具体功能时提及。

NNI 具有高扩展性，高级用户可以定制自己的 Tuner、 Assessor，以及训练平台
来适应不同的需求。

..  toctree::
    :maxdepth: 2

    实现 Trial<./TrialExample/Trials>
    Tuners <builtin_tuner>
    Assessors <builtin_assessor>
    训练平台 <training_services>
    示例 <examples>
    Web 界面 <Tutorial/WebUI>
    如何调试 <Tutorial/HowToDebug>
    高级 <hpo_advanced>