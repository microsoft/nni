#################
Tuners
#################

NNI provides an easy way to adopt an approach to set up parameter tuning algorithms, we call them **Tuner**.

Tuner receives metrics from `Trial` to evaluate the performance of a specific parameters/architecture configures. And tuner sends next hyper-parameter or architecture configure to Trial.

In NNI, we support two approaches to set the tuner: first is directly use builtin tuner provided by nni sdk, second is customize a tuner file by yourself. We also have Advisor that combines the functinality of Tuner & Assessor.

For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Builtin Tuners<BuiltinTuner>
    Customized Tuners<CustomizeTuner>
    Customized Advisor<CustomizeAdvisor>