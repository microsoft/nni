NNI 中的 PBTTuner
===

## PBTTuner

Population Based Training (PBT，基于种群的训练) 来自于 [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846v1)。 它是一种简单的异步优化算法，在固定的计算资源下，它能有效的联合优化一组模型及其超参来最大化性能。 重要的是，PBT 探索的是超参设置的规划，而不是通过整个训练过程中，来试图找到某个固定的参数配置。

PBTTuner 通过几次 Trial 来初始化种群。 用户可以设置特定数量的训练 Epoch。 在一定数量的 Epoch 后， 参数和超参指标较差的 Trial 会被较好的 Trial 替换掉（即，对好的结果进行更多的挖掘）。 然后超参会被扰动（探索）。

在 NNI 的实现中，Trial 代码中训练的 Epoch 对应于 PBT 的步骤，这与其它 Tuner 不同。 在每个步骤结束时，PBT Tuner 会进行挖掘和探索 - 用新的 Trial 替换掉某些 Trial。 这通过不断修改 `load_checkpoint_dir` 和 `save_checkpoint_dir` 的值来实现的。 可通过直接修改 `load_checkpoint_dir` 来替换参数和超参，并用 `save_checkpoint_dir` 来保存下一步会被读取的检查点。 为此，需要能让所有 Trial 访问的共享文件夹。

如果 Experiment 在本机模式下运行，用户可提供 `all_checkpoint_dir` 参数来定义 `load_checkpoint_dir` 和 `save_checkpoint_dir` 的上级目录 (`checkpoint_dir` 会是 `all_checkpoint_dir/<population-id>/<step>`)。 默认情况下，`all_checkpoint_dir` 会设为 `~/nni/experiments/<exp-id>/checkpoint`。 如果 Experiment 不是本机模式，用户需要提供共享存储文件夹 `all_checkpoint_dir` 来让所有工作计算机都能访问（运行 Tuner 的计算机不需要访问此文件夹）。
