## Create multi-phase experiment

Typically each trial job gets a single configuration (e.g., hyperparameters) from tuner, tries this configuration and reports result, then exits. But sometimes a trial job may wants to request multiple configurations from tuner. We find this is a very compelling feature. For example:

1. Job launch takes tens of seconds in some training platform. If a configuration takes only around a minute to finished, running only one configuration in a trial job would be every inefficient. An appealing way is that a trial job requests a configuration and finished it, then requests another configuration and run. The extreme case is that a trial job can run infinite configurations. If you set concurrency to be for example 6, there would be 6 __long running__ jobs keeping trying different configurations.

2. Some types of models have to be trained phase by phase, the configuration of next phase depends on the results of previous phase(s). For example, to find the best quantization for a model, the training procedure is often as follows: the auto-quantization algorithm (i.e., tuner in NNI) chooses a size of bits (e.g., 16 bits), a trial job gets this configuration and trains the model for some epochs and reports result (e.g., accuracy). The algorithm receives this result and makes decision of changing 16 bits to 8 bits, or changing back to 32 bits. This process is repeated for a configured times.

The above cases can be supported by the same feature, i.e., multi-phase execution. To support those cases, basically a trial job should be able to request multiple configurations from tuner. Tuner is aware of whether two configuration requests are from the same trial job or different ones. Also in multi-phase a trial job can report multiple final results. is there restriction on sequence???

__To use multi-phase experiment, please follow below steps:__

1. Implement nni.multi_phase.MultiPhaseTuner. For example, this [ENAS tuner](https://github.com/countif/enas_nni/blob/master/nni/examples/tuners/enas/nni_controller_ptb.py) is a multi-phase Tuner which implements nni.multi_phase.MultiPhaseTuner. While implementing your MultiPhaseTuner, you may want to use the trial_job_id parameter of generate_parameters method to generate hyper parameters for each trial job.

1. Set `multiPhase` field to `true`, and configure your tuner implemented in step 1 as customized tuner in configuration file, for example:

    ```yaml
    ...
    multiPhase: true
    tuner:
      codeDir: tuners/enas
      classFileName: nni_controller_ptb.py
      className: ENASTuner
      classArgs:
        say_hello: "hello"
    ...
    ```

1. Invoke nni.get_next_parameter() API for multiple times as needed in a trial, for example:

    ```python
    for i in range(5):
        # get parameter from tuner
        tuner_param = nni.get_next_parameter()

        # consume the params
        # ...
        # report final result somewhere for the parameter retrieved above
        nni.report_final_result()
        # ...
    ```
