## 多阶段 Experiment

通常，每个 Trial 任务只需要从 Tuner 获取一个配置（超参等），然后使用这个配置执行并报告结果，然后退出。 但有时，一个 Trial 任务可能需要从 Tuner 请求多次配置。 这是一个非常有用的功能。 例如：

1. 在一些训练平台上，需要数十秒来启动一个任务。 如果一个配置只需要一分钟就能完成，那么每个 Trial 任务中只运行一个配置就会非常低效。 这种情况下，可以在同一个 Trial 任务中，完成一个配置后，再请求并完成另一个配置。 极端情况下，一个 Trial 任务可以运行无数个配置。 如果设置了并发（例如设为 6），那么就会有 6 个**长时间**运行的任务来不断尝试不同的配置。

2. 有些类型的模型需要进行多阶段的训练，而下一个阶段的配置依赖于前一个阶段的结果。 例如，为了找到模型最好的量化结果，训练过程通常为：自动量化算法（例如 NNI 中的 TunerJ）选择一个位宽（如 16 位）， Trial 任务获得此配置，并训练数个 epoch，并返回结果（例如精度）。 算法收到结果后，决定是将 16 位改为 8 位，还是 32 位。 此过程会重复多次。

上述情况都可以通过多阶段执行的功能来支持。 为了支持这些情况，一个 Trial 任务需要能从 Tuner 请求多个配置。 Tuner 需要知道两次配置请求是否来自同一个 Trial 任务。 同时，多阶段中的 Trial 任务需要多次返回最终结果。

## 创建多阶段的 Experiment

### 实现使用多阶段的 Trial 代码：

**1. 更新 Trial 代码**

Trial 代码中使用多阶段非常容易，示例如下：

```python
# ...
for i in range(5):
    # 从 Tuner 中获得参数
    tuner_param = nni.get_next_parameter()
    # 如果没有更多超参可生成，nni.get_next_parameter 会返回 None。
    if tuner_param is None:
      break

    # 使用参数
    # ...
    # 返回最终结果
        nni.report_final_result()
        # ...
# ...
```

在多阶段 Experiment 中，每次 API `nni.get_next_parameter()` 被调用时，会返回 Tuner 新生成的超参，然后 Trial 代码会使用新的超参，并返回其最终结果。 `nni.get_next_parameter()` 和 `nni.report_final_result()` 需要依次被调用：**先调用前者，然后调用后者，并按此顺序重复调用**。 如果 `nni.get_next_parameter()` 被连续多次调用，然后再调用 `nni.report_final_result()`，这会造成最终结果只会与 get_next_parameter 所返回的最后一个配置相关联。 因此，前面的 get_next_parameter 调用都没有关联的结果，这可能会造成一些多阶段算法出问题。

注意，如果 `nni.get_next_parameter` 返回 None，表示 Tuner 没有生成更多的超参。

**2. Experiment 配置**

要启用多阶段，需要在 Experiment 的 YAML 配置文件中增加 `multiPhase: true`。 如果不添加此参数，`nni.get_next_parameter()` 会一直返回同样的配置。

多阶段 Experiment 配置示例：

```yaml
authorName: default
experimentName: multiphase experiment
trialConcurrency: 2
maxExecDuration: 1h
maxTrialNum: 8
trainingServicePlatform: local
searchSpacePath: search_space.json
multiPhase: true
useAnnotation: false
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python3 mytrial.py
  codeDir: .
  gpuNum: 0
```

### 实现使用多阶段的 Tuner：

强烈建议首先阅读[自定义 Tuner](https://nni.readthedocs.io/zh/latest/Tuner/CustomizeTuner.html)，再开始实现多阶段 Tuner。 与普通 Tuner 一样，需要从 `Tuner` 类继承。 当通过配置启用多阶段时（将 `multiPhase` 设为 true），Tuner 会通过下列方法得到一个新的参数 `trial_job_id`：

```text
generate_parameters
generate_multiple_parameters
receive_trial_result
receive_customized_trial_result
trial_end
```

有了这个信息， Tuner 能够知道哪个 Trial 在请求配置信息， 返回的结果是哪个 Trial 的。 通过此信息，Tuner 能够灵活的为不同的 Trial 及其阶段实现功能。 例如，可在 generate_parameters 方法中使用 trial_job_id 来为特定的 Trial 任务生成超参。

### 支持多阶段 Experiment 的 Tuner：

[TPE](../Tuner/HyperoptTuner.md), [Random](../Tuner/HyperoptTuner.md), [Anneal](../Tuner/HyperoptTuner.md), [Evolution](../Tuner/EvolutionTuner.md), [SMAC](../Tuner/SmacTuner.md), [NetworkMorphism](../Tuner/NetworkmorphismTuner.md), [MetisTuner](../Tuner/MetisTuner.md), [BOHB](../Tuner/BohbAdvisor.md), [Hyperband](../Tuner/HyperbandAdvisor.md).

### 支持多阶段 Experiment 的训练平台：

[本机](../TrainingService/LocalMode.md), [远程计算机](../TrainingService/RemoteMachineMode.md), [OpenPAI](../TrainingService/PaiMode.md)