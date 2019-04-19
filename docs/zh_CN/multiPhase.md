## 多阶段 Experiment

通常，每个 Trial 任务只需要从 Tuner 获取一个配置（超参等），然后使用这个配置执行并报告结果，然后退出。 但有时，一个 Trial 任务可能需要从 Tuner 请求多次配置。 这是一个非常有用的功能。 例如：

1. 在一些训练平台上，需要数十秒来启动一个任务。 如果一个配置只需要一分钟就能完成，那么每个 Trial 任务中只运行一个配置就会非常低效。 这种情况下，可以在同一个 Trial 任务中，完成一个配置后，再请求并完成另一个配置。 极端情况下，一个 Trial 任务可以运行无数个配置。 如果设置了并发（例如设为 6），那么就会有 6 个**长时间**运行的任务来不断尝试不同的配置。

2. Some types of models have to be trained phase by phase, the configuration of next phase depends on the results of previous phase(s). For example, to find the best quantization for a model, the training procedure is often as follows: the auto-quantization algorithm (i.e., tuner in NNI) chooses a size of bits (e.g., 16 bits), a trial job gets this configuration and trains the model for some epochs and reports result (e.g., accuracy). The algorithm receives this result and makes decision of changing 16 bits to 8 bits, or changing back to 32 bits. This process is repeated for a configured times.

The above cases can be supported by the same feature, i.e., multi-phase execution. To support those cases, basically a trial job should be able to request multiple configurations from tuner. Tuner is aware of whether two configuration requests are from the same trial job or different ones. Also in multi-phase a trial job can report multiple final results.

Note that, `nni.get_next_parameter()` and `nni.report_final_result()` should be called sequentially: **call the former one, then call the later one; and repeat this pattern**. If `nni.get_next_parameter()` is called multiple times consecutively, and then `nni.report_final_result()` is called once, the result is associated to the last configuration, which is retrieved from the last get_next_parameter call. So there is no result associated to previous get_next_parameter calls, and it may cause some multi-phase algorithm broken.

## Create multi-phase experiment

### Write trial code which leverages multi-phase:

**1. Update trial code**

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
    

**2. Modify experiment configuration**

To enable multi-phase, you should also add `multiPhase: true` in your experiment yaml configure file. If this line is not added, `nni.get_next_parameter()` would always return the same configuration. For all the built-in tuners/advisors, you can use multi-phase in your trial code without modification of tuner/advisor spec in the yaml configure file.

### Write a tuner that leverages multi-phase:

Before writing a multi-phase tuner, we highly suggest you to go through [Customize Tuner](https://nni.readthedocs.io/en/latest/Customize_Tuner.html). Different from writing a normal tuner, your tuner needs to inherit from `MultiPhaseTuner` (in nni.multi_phase_tuner). The key difference between `Tuner` and `MultiPhaseTuner` is that the methods in MultiPhaseTuner are aware of additional information, that is, `trial_job_id`. With this information, the tuner could know which trial is requesting a configuration, and which trial is reporting results. This information provides enough flexibility for your tuner to deal with different trials and different phases. For example, you may want to use the trial_job_id parameter of generate_parameters method to generate hyperparameters for a specific trial job.

Of course, to use your multi-phase tuner, **you should add `multiPhase: true` in your experiment yaml configure file**.

[ENAS tuner](https://github.com/countif/enas_nni/blob/master/nni/examples/tuners/enas/nni_controller_ptb.py) is an example of a multi-phase tuner.