## 创建多阶段的 Experiment

通常情况下，每个 Trial 作业只从 Tuner 获得一组配置（如超参），然后运行 Experiment。也就是说，通过这组超参来训练模型，并返回结果给 Tuner。 有时候，可能需要在一个 Trial 作业中训练多个模型，并在它们之间共享信息，或者通过创建更少的 Trial 任务来节省资源。例如：

1. 在一个 Trial 作业中依次训练多个模型。这样，后面的模型可以利用先前模型的权重和其它信息，并可以使用不同的超参组合。
2. 在有限的资源上训练大量的模型，将多个模型放到一个 Trial 作业中训练，能够节约系统创建 Trial 作业的时间。
3. 还有的情况，希望在一个 Trial 任务中训练多个需要不同超参的模型。注意，如果为一个 Trial 作业分配多个 GPU，并且会并发训练模型，需要在代码中正确分配 GPU 资源。

在上述情况中，可利用 NNI 的多阶段 Experiment 来在同一个 Trial 任务中训练具有不同超参的多个模型。

多阶段 Experiment，是指 Trial 作业会从 Tuner 请求多次超参，并多次返回最终结果。

参考以下步骤来使用多阶段 Experiment：

1. 实现 nni.multi_phase.MultiPhaseTuner。 如，[ENAS tuner](https://github.com/countif/enas_nni/blob/master/nni/examples/tuners/enas/nni_controller_ptb.py) 是实现了 nni.multi_phase.MultiPhaseTuner 的 Tuner。 在实现多阶段 Tuner 时，可能要用 generate_parameters 中的 trial_job_id 参数来为每个 Trial 作业生成超参。

2. 设置 `multiPhase` 的值为 `true`，并将第一步中实现的 Tuner 作为自定义 Tuner 进行配置，例如：
    
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

3. 根据需要，在 Trial 代码中可多次调用 nni.get_next_parameter() API，例如：
    
    ```python
    for i in range(5):
        # 从 Tuner 中获得参数
        tuner_param = nni.get_next_parameter()
    
        # 使用参数
        # ...
        # 为上面获取的参数返回最终结果
        nni.report_final_result()
        # ...
    ```