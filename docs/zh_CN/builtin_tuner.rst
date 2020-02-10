内置 Tuner
==============

NNI provides an easy way to adopt an approach to set up parameter tuning algorithms, we call them **Tuner**.

Tuner receives metrics from `Trial` to evaluate the performance of a specific parameters/architecture configures. And tuner sends next hyper-parameter or architecture configure to Trial.


..  toctree::
    :maxdepth: 1
    
    概述<Tuner/BuiltinTuner>
    TPE<Tuner/HyperoptTuner>
    Random Search（随机搜索）<Tuner/HyperoptTuner>
    Anneal（退火）<Tuner/HyperoptTuner>
    Naïve Evolution（朴素进化）<Tuner/EvolutionTuner>
    SMAC<Tuner/SmacTuner>
    Metis Tuner<Tuner/MetisTuner>
    Batch Tuner（批处理）<Tuner/BatchTuner>
    Grid Search（遍历）<Tuner/GridsearchTuner>
    GP Tuner<Tuner/GPTuner>
    Network Morphism<Tuner/NetworkmorphismTuner>
    Hyperband<Tuner/HyperbandAdvisor>
    BOHB<Tuner/BohbAdvisor>
    PPO Tuner <Tuner/PPOTuner>
