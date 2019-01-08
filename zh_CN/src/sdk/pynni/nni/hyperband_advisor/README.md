# NNI 中使用 Hyperband

## 1. 介绍

[Hyperband](https://arxiv.org/pdf/1603.06560.pdf) 是一种流行的自动机器学习算法。 Hyperband 的基本思想是对配置分组，每组有 `n` 个随机生成的超参配置，每个配置使用 `r` 次资源（如，epoch 数量，批处理数量等）。 当 `n` 个配置完成后，会选择最好的 `n/eta` 个配置，并增加 `r*eta` 次使用的资源。 最后，会选择出的最好配置。

## 2. 实现并行

首先，此样例是基于 MsgDispatcherBase 来实现的自动机器学习算法，而不是基于调参器和评估器。 使用这种实现方法，是因为 Hyperband 集成了调参器和评估器两者的函数，因而，将它叫做 Advisor。

其次，本实现完全利用了 Hyperband 内部的并行性。 具体来说，下一个分组不会严格的在当前分组结束后再运行，只要有资源，就可以开始运行新的分组。

## 3. 用法

要使用 Hyperband，需要在实验的 yaml 配置文件进行如下改动。

    advisor:
      #可选项: Hyperband
      builtinAdvisorName: Hyperband
      classArgs:
        #R: 最大的步骤
        R: 100
        #eta: 丢弃的尝试的比例
        eta: 3
        #可选项: maximize, minimize
        optimize_mode: maximize
    

注意，一旦使用了 Advisor，就不能在配置文件中添加调参器和评估器。 If you use Hyperband, among the hyperparameters (i.e., key-value pairs) received by a trial, there is one more key called `STEPS` besides the hyperparameters defined by user. By using this `STEPS`, the trial can control how long it runs.

`R` and `eta` are the parameters of Hyperband that you can change. `R` means the maximum STEPS that can be allocated to a configuration. Here, STEPS could mean the number of epochs or mini-batches. This `STEPS` should be used by the trial to control how long it runs. Refer to the example under `examples/trials/mnist-hyperband/` for details.

`eta` means `n/eta` configurations from `n` configurations will survive and rerun using more STEPS.

Here is a concrete example of `R=81` and `eta=3`:

|   | s=4  | s=3  | s=2  | s=1  | s=0  |
| - | ---- | ---- | ---- | ---- | ---- |
| i | n r  | n r  | n r  | n r  | n r  |
| 0 | 81 1 | 27 3 | 9 9  | 6 27 | 5 81 |
| 1 | 27 3 | 9 9  | 3 27 | 2 81 |      |
| 2 | 9 9  | 3 27 | 1 81 |      |      |
| 3 | 3 27 | 1 81 |      |      |      |
| 4 | 1 81 |      |      |      |      |

`s` means bracket, `n` means the number of configurations that are generated, the corresponding `r` means how many STEPS these configurations run. `i` means round, for example, bracket 4 has 5 rounds, bracket 3 has 4 rounds.

About how to write trial code, please refer to the instructions under `examples/trials/mnist-hyperband/`.

## 4. To be improved

The current implementation of Hyperband can be further improved by supporting simple early stop algorithm, because it is possible that not all the configurations in the top `n/eta` perform good. The unpromising configurations can be stopped early.

In the current implementation, configurations are generated randomly, which follows the design in the [paper](https://arxiv.org/pdf/1603.06560.pdf). To further improve, configurations could be generated more wisely by leveraging advanced algorithms.