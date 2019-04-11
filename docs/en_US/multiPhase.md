## Create multi-phase experiment

Typically each trial job gets a single configuration (e.g., hyperparameters) from tuner, tries this configuration and reports result, then exits. But sometimes a trial job may wants to request multiple configurations from tuner. We find this is a very compelling feature. For example:

1. Job launch takes tens of seconds in some training platform. If a configuration takes only around a minute to finish, running only one configuration in a trial job would be every inefficient. An appealing way is that a trial job requests a configuration and finishes it, then requests another configuration and run. The extreme case is that a trial job can run infinite configurations. If you set concurrency to be for example 6, there would be 6 __long running__ jobs keeping trying different configurations.

2. Some types of models have to be trained phase by phase, the configuration of next phase depends on the results of previous phase(s). For example, to find the best quantization for a model, the training procedure is often as follows: the auto-quantization algorithm (i.e., tuner in NNI) chooses a size of bits (e.g., 16 bits), a trial job gets this configuration and trains the model for some epochs and reports result (e.g., accuracy). The algorithm receives this result and makes decision of changing 16 bits to 8 bits, or changing back to 32 bits. This process is repeated for a configured times.

The above cases can be supported by the same feature, i.e., multi-phase execution. To support those cases, basically a trial job should be able to request multiple configurations from tuner. Tuner is aware of whether two configuration requests are from the same trial job or different ones. Also in multi-phase a trial job can report multiple final results. 

Note that, currently multi-phase only supports one pattern of calling `nni.get_next_parameter()` and `nni.report_final_result()`: __call the former one, then call the later one; and repeat this pattern__. If calling `nni.get_next_parameter()` multiple times consecutively, calling `nni.report_final_result()` after that is reporting the final result of the latest retrieved configuration by `nni.get_next_parameter()`.

__Write trial code which leverages multi-phase:__

It is pretty simple to use multi-phase in trial code, an example is shown below:

    ```python
    # ...
    for i in range(5):
        # get parameter from tuner
        tuner_param = nni.get_next_parameter()

        # consume the params
        # ...
        # report final result somewhere for the parameter retrieved above
        nni.report_final_result()
        # ...
    # ...
    ```
To enable multi-phase, you should also add `multiPhase: true` in your experiment yaml configure file. If this line is not added, `nni.get_next_parameter()` would always return the same configuration. For all the built-in tuners/advisors, you can use multi-phase in your trial code without modification of tuner/advisor spec in the yaml configure file.

__Write a tuner that leverages multi-phase:__

Before writing a multi-phase tuner, we highly suggest you to go through  [Customize Tuner](https://nni.readthedocs.io/en/latest/Customize_Tuner.html). Different from writing a normal tuner, your tuner needs to inherit from `MultiPhaseTuner` (in nni.multi_phase_tuner). The key difference between `Tuner` and `MultiPhaseTuner` is that the methods in MultiPhaseTuner are aware of additional information, that is, `trial_job_id`. With this information, the tuner could know which trial is requesting a configuration, and which trial is reporting results. This information provides enough flexibility for your tuner to deal with different trials and different phases. For example, you may want to use the trial_job_id parameter of generate_parameters method to generate hyperparameters for a specific trial job.

Of course, to use your multi-phase tuner, you should add `multiPhase: true` in your experiment yaml configure file.

[ENAS tuner](https://github.com/countif/enas_nni/blob/master/nni/examples/tuners/enas/nni_controller_ptb.py) is an example of a multi-phase tuner.
