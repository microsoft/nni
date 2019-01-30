#################
Tuner（调参器）
#################

NNI 能用简单快速的方法来配置超参调优算法，称之为 **Tuner**。

Tuner receives the result from `Trial` as a matrix to evaluate the performance of a specific parameters/architecture configures. And tuner sends next hyper-parameter or architecture configure to Trial.

In NNI, we support two approaches to set the tuner: first is directly use builtin tuner provided by nni sdk, second is customize a tuner file by yourself. We also have Advisor that combines the functinality of Tuner & Assessor.

For details, please refer to the following tutorials:

..  toctree::
    Builtin Tuners<Builtin_Tuner>
    Customized Tuners<Customize_Tuner>
    Customized Advisor<Customize_Advisor>