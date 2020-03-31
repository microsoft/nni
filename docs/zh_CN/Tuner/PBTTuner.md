NNI 中的 PBTTuner
===

## PBTTuner

Population Based Training (PBT，基于种群的训练) 来自于 [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846v1)。 它是一种简单的异步优化算法，在固定的计算资源下，它能有效的联合优化一组模型及其超参来最大化性能。 重要的是，PBT 探索的是超参设置的规划，而不是通过整个训练过程中，来试图找到某个固定的参数配置。

PBTTuner 通过几次 Trial 来初始化种群。 用户可以设置特定数量的训练 Epoch。 在一定数量的 Epoch 后， 参数和超参指标较差的 Trial 会被较好的 Trial 替换掉（即，对好的结果进行更多的挖掘）。 然后超参会被扰动（探索）。

在 NNI 的实现中，Trial 代码中训练的 Epoch 对应于 PBT 的步骤，这与其它 Tuner 不同。 在每个步骤结束时，PBT Tuner 会进行挖掘和探索 - 用新的 Trial 替换掉某些 Trial。 这通过不断修改 `load_checkpoint_dir` 和 `save_checkpoint_dir` 的值来实现的。 可通过直接修改 `load_checkpoint_dir` 来替换参数和超参，并用 `save_checkpoint_dir` 来保存下一步会被读取的检查点。 为此，需要能让所有 Trial 访问的共享文件夹。

If the experiment is running in local mode, users could provide an argument `all_checkpoint_dir` which will be the base folder of `load_checkpoint_dir` and `save_checkpoint_dir` (`checkpoint_dir` is set to `all_checkpoint_dir/<population-id>/<step>`). By default, `all_checkpoint_dir` is set to be `~/nni/experiments/<exp-id>/checkpoint`. If the experiment is in non-local mode, then users should provide a path in a shared storage folder which is mounted at `all_checkpoint_dir` on worker machines (but it's not necessarily available on the machine which runs tuner).
