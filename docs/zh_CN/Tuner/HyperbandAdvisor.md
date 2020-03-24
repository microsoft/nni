# NNI 中使用 Hyperband

## 1. 介绍

[Hyperband](https://arxiv.org/pdf/1603.06560.pdf) is a popular autoML algorithm. The basic idea of Hyperband is to create several buckets, each having `n` randomly generated hyperparameter configurations, each configuration using `r` resources (e.g., epoch number, batch number). After the `n` configurations are finished, it chooses the top `n/eta` configurations and runs them using increased `r*eta` resources. 最后，会选择出的最好配置。

## 2. Implementation with full parallelism

First, this is an example of how to write an autoML algorithm based on MsgDispatcherBase, rather than Tuner and Assessor. Hyperband is implemented in this way because it integrates the functions of both Tuner and Assessor, thus, we call it Advisor.

其次，本实现完全利用了 Hyperband 内部的并行性。 Specifically, the next bucket is not started strictly after the current bucket. Instead, it starts when there are available resources.

## 3. 用法

要使用 Hyperband，需要在 Experiment 的 YAML 配置文件进行如下改动。

    advisor:
      #可选项: Hyperband
      builtinAdvisorName: Hyperband
      classArgs:
        #R: 最大的步骤
        R: 100
        #eta: 丢弃的 Trial 的比例
        eta: 3
        #可选项: maximize, minimize
        optimize_mode: maximize
    

Note that once you use Advisor, you are not allowed to add a Tuner and Assessor spec in the config file. If you use Hyperband, among the hyperparameters (i.e., key-value pairs) received by a trial, there will be one more key called `TRIAL_BUDGET` defined by user. **使用 `TRIAL_BUDGET`，Trial 能够控制其运行的时间。</p> 

对于 Trial 代码中 `report_intermediate_result(metric)` 和 `report_final_result(metric)` 的**`指标` 应该是数值，或者用一个 dict，并保证其中有键值为 default 的项目，其值也为数值型**。 这是需要进行最大化或者最小化优化的数值，如精度或者损失度。

`R` 和 `eta` 是 Hyperband 中可以改动的参数。 `R` 表示可以分配给 Trial 的最大资源。 这里，资源可以代表 epoch 或 批处理数量。 `TRIAL_BUDGET` 应该被尝试代码用来控制运行的次数。 参考示例 `examples/trials/mnist-advisor/` ，了解详细信息。

`eta` 表示 `n` 个配置中的 `n/eta` 个配置会留存下来，并用更多的资源来运行。

下面是 `R=81` 且 `eta=3` 时的示例：

|   | s=4  | s=3  | s=2  | s=1  | s=0  |
| - | ---- | ---- | ---- | ---- | ---- |
| i | n r  | n r  | n r  | n r  | n r  |
| 0 | 81 1 | 27 3 | 9 9  | 6 27 | 5 81 |
| 1 | 27 3 | 9 9  | 3 27 | 2 81 |      |
| 2 | 9 9  | 3 27 | 1 81 |      |      |
| 3 | 3 27 | 1 81 |      |      |      |
| 4 | 1 81 |      |      |      |      |

`s` 表示分组， `n` 表示生成的配置数量，相应的 `r` 表示配置使用多少资源来运行。 `i` 表示轮数，如分组 4 有 5 轮，分组 3 有 4 轮。

For information about writing trial code, please refer to the instructions under `examples/trials/mnist-hyperband/`.

## 4. Future improvements

The current implementation of Hyperband can be further improved by supporting a simple early stop algorithm since it's possible that not all the configurations in the top `n/eta` perform well. Any unpromising configurations should be stopped early.

In the current implementation, configurations are generated randomly which follows the design in the [paper](https://arxiv.org/pdf/1603.06560.pdf). As an improvement, configurations could be generated more wisely by leveraging advanced algorithms.