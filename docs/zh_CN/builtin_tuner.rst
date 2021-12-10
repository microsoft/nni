.. fdf455f855f2beb05b538d36ac1d2c51

内置 Tuner
==============

NNI 能用简单快速的方法来配置超参调优算法，称之为 **Tuner**。

Tuner 从 Trial 接收指标结果，来评估一组超参或网络结构的性能。 然后 Tuner 会将下一组超参或网络结构的配置发送给新的 Trial。


..  toctree::
    :maxdepth: 1
    
    概述<Tuner/BuiltinTuner>
    Random Search（随机搜索）<Tuner/HyperoptTuner>
    Naïve Evolution（朴素进化）<Tuner/EvolutionTuner>
    SMAC<Tuner/SmacTuner>
    Metis Tuner<Tuner/MetisTuner>
    Batch Tuner（批处理）<Tuner/BatchTuner>
    Grid Search（遍历）<Tuner/GridsearchTuner>
    GP Tuner<Tuner/GPTuner>
    Network Morphism<Tuner/NetworkmorphismTuner>
    Hyperband<Tuner/HyperbandAdvisor>
    BOHB<Tuner/BohbAdvisor>
    PBT Tuner <Tuner/PBTTuner>
    DNGO Tuner <Tuner/DngoTuner>
