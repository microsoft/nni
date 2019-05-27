#################
Tuner（调参器）
#################

NNI 能用简单快速的方法来配置超参调优算法，称之为 **Tuner**。

Tuner 从 Trial 接收指标结果，来评估一组超参或网络结构的性能。 然后 Tuner 会将下一组超参或网络结构的配置发送给新的 Trial。

在 NNI 中，有两种方法来选择调优算法：可以使用内置的 Tuner，也可以自定义 Tuner。 另外，也可以使用 Advisor，它同时支持 Tuner 和 Assessor 的功能。

详细信息，参考以下教程：

..  toctree::
    :maxdepth: 2

    内置 Tuner<BuiltinTuner>
    自定义 Tuner<CustomizeTuner>
    自定义 Advisor<CustomizeAdvisor>