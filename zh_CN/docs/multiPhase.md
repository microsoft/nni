## 创建多阶段的实验

通常情况下，每个尝试作业只从调参器获得一组配置（如超参），然后运行实验。也就是说，通过这组超参来训练模型，并返回结果给调参器。 有时候，可能需要在一个尝试作业中训练多个模型，并在它们之间共享信息，或者通过创建更少的尝试任务来节省资源。例如：

1. 在一个尝试作业中依次训练多个模型。这样，后面的模型可以利用先前模型的权重和其它信息，并可以使用不同的超参组合。
2. Train large amount of models on limited system resource, combine multiple models together to save system resource to create large amount of trial jobs.
3. Any other scenario that you would like to train multiple models with different hyper parameters in one trial job, be aware that if you allocate multiple GPUs to a trial job and you train multiple models concurrently within on trial job, you need to allocate GPU resource properly by your trial code.

In above cases, you can leverage NNI multi-phase experiment to train multiple models with different hyper parameters within each trial job.

Multi-phase experiments refer to experiments whose trial jobs request multiple hyper parameters from tuner and report multiple final results to NNI.

To use multi-phase experiment, please follow below steps:

1. Implement nni.multi_phase.MultiPhaseTuner. For example, this [ENAS tuner](https://github.com/countif/enas_nni/blob/master/nni/examples/tuners/enas/nni_controller_ptb.py) is a multi-phase Tuner which implements nni.multi_phase.MultiPhaseTuner. While implementing your MultiPhaseTuner, you may want to use the trial_job_id parameter of generate_parameters method to generate hyper parameters for each trial job.

2. Set ```multiPhase``` field to ```true```, and configure your tuner implemented in step 1 as customized tuner in configuration file, for example:

```yml
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

3. Invoke nni.get_next_parameter() API for multiple times as needed in a trial, for example:

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