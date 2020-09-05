# 经典 NAS 算法

在经典 NAS 算法中，每个结构都作为 Trial 来训练，而 NAS 算法来充当 Tuner。 因此，训练过程能使用 NNI 中的超参调优框架，Tuner 为下一个 Trial 生成新的结构，Trial 在训练平台中运行。

## 快速入门

下例展示了如何使用经典 NAS 算法。 与 NNI 超参优化非常相似。

```python
model = Net()

# 从 Tuner 中获得选择的架构，并应用到模型上
get_and_apply_next_architecture(model)
train(model)  # 训练模型的代码
acc = test(model)  # 测试训练好的模型
nni.report_final_result(acc)  # 报告所选架构的性能
```

首先，实例化模型。 模型中，搜索空间通过 `LayerChoice` 和 `InputChoice` 来定义。 然后，调用 `get_and_apply_next_architecture(model)` 来获得特定的结构。 此函数会从 Tuner （即，经典的 NAS 算法）中接收结构，并应用到 `model` 上。 此时，`model` 成为了某个结构，不再是搜索空间。 然后可以像普通 PyTorch 模型一样训练此模型。 获得模型精度后，调用 `nni.report_final_result(acc)` 来返回给 Tuner。

至此，Trial 代码已准备好了。 然后，准备好 NNI 的 Experiment，即搜索空间文件和 Experiment 配置文件。 与 NNI 超参优化不同的是，要通过运行命令（详情参考[这里](../Tutorial/Nnictl.md)）从 Trial 代码中自动生成搜索空间文件。

`nnictl ss_gen --trial_command="运行 Trial 代码的命令"`

此命令会自动生成 `nni_auto_gen_search_space.json` 文件。 然后，将生成的搜索空间文件路径填入 Experiment 配置文件的 `searchSpacePath` 字段。 配置文件中的其它字段，可参考[此教程](../Tutorial/QuickStart.md)。

目前，经典 NAS 仅支持 [PPO Tuner](../Tuner/BuiltinTuner.md) 和 [随机 Tuner ](https://github.com/microsoft/nni/tree/master/examples/tuners/random_nas_tuner)。 未来将支持更多经典 NAS 算法。

完整的 [PyTorch 示例](https://github.com/microsoft/nni/tree/master/examples/nas/classic_nas)，以及 [TensorFlow 示例](https://github.com/microsoft/nni/tree/master/examples/nas/classic_nas-tf)。

## 用于调试的独立模式

为了便于调试，其支持独立运行模式，可直接运行 Trial 命令，而不启动 NNI Experiment。 可以通过此方法来检查 Trial 代码是否可正常运行。 在独立模式下，`LayerChoice` 和 `InputChoice` 会选择第一个的候选项。