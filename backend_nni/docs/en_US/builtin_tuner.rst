Builtin-Tuners
==============

NNI provides an easy way to adopt an approach to set up parameter tuning algorithms, we call them **Tuner**.

Tuner receives metrics from `Trial` to evaluate the performance of a specific parameters/architecture configuration. Tuner sends the next hyper-parameter or architecture configuration to Trial.


..  toctree::
    :maxdepth: 1
    
    Overview <Tuner/BuiltinTuner>
    TPE <Tuner/HyperoptTuner>
    Random Search <Tuner/HyperoptTuner>
    Anneal <Tuner/HyperoptTuner>
    Naive Evolution <Tuner/EvolutionTuner>
    SMAC <Tuner/SmacTuner>
    Metis Tuner <Tuner/MetisTuner>
    Batch Tuner <Tuner/BatchTuner>
    Grid Search <Tuner/GridsearchTuner>
    GP Tuner <Tuner/GPTuner>
    Network Morphism <Tuner/NetworkmorphismTuner>
    Hyperband <Tuner/HyperbandAdvisor>
    BOHB <Tuner/BohbAdvisor>
    PPO Tuner <Tuner/PPOTuner>
    PBT Tuner <Tuner/PBTTuner>
