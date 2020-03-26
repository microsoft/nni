#############################
自动（超参数）调优
#############################

自动调优是 NNI 提供的关键功能之一，主要应用场景是
超参调优。 Tuning specifically applies to trial code. We provide a lot of popular
自动调优算法（称为 Tuner ）和一些提前终止算法（称为 Assessor）。
NNI supports running trials on various training platforms, for example, on a local machine,
on several servers in a distributed manner, or on platforms such as OpenPAI, Kubernetes, etc.

NNI 的其它重要功能，例如模型压缩，特征工程，也可以进一步
enhanced by auto tuning, which we'll described when introducing those features.

NNI has high extensibility, advanced users can customize their own Tuner, Assessor, and Training Service
根据自己的需求。

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