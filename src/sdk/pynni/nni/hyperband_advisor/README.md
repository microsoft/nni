Hyperband on nni
===

## 1. Introduction
[Hyperband][1] is a popular automl algorithm. The basic idea of Hyperband is that it creates several buckets, each bucket has `n` randomly generated hyperparameter configurations, each configuration uses `r` resource (e.g., epoch number, batch number). After the `n` configurations is finished, it chooses top `n/eta` configurations and runs them using increased `r*eta` resource. At last, it chooses the best configuration it has found so far.

## 2. Implementation with fully parallelism
Frist, this is an example of how to write an automl algorithm based on MsgDispatcherBase, rather than Tuner and Assessor. Hyperband is implemented in this way because it integrates the functions of both Tuner and Assessor, thus, we call it advisor.

Second, this implementation fully leverages Hyperband's internal parallelism. More specifically, the next bucket is not started strictly after the current bucket, instead, it starts when there is available resource.

## 3. Usage
To use Hyperband, you should add the following spec in your experiment's yaml config file:

```
advisor:
  #choice: Hyperband
  builtinAdvisorName: Hyperband
  classArgs:
    #R: the maximum STEPS
    R: 100
    #eta: proportion of discarded trials
    eta: 3
    #choice: maximize, minimize
    optimize_mode: maximize
```

Note that once you use advisor, it is not allowed to add tuner and assessor spec in the config file any more.
If you use Hyperband, among the hyperparameters (i.e., key-value pairs) received by a trial, there is one more key called `STEPS` besides the hyperparameters defined by user. **By using this `STEPS`, the trial can control how long it runs**.

For `report_intermediate_result(metric)` and `report_final_result(metric)` in your trial code, **`metric` should be either a number or a dict which has a key `default` with a number as its value**. This number is the one you want to maximize or minimize, for example, accuracy or loss.

`R` and `eta` are the parameters of Hyperband that you can change. `R` means the maximum STEPS that can be allocated to a configuration. Here, STEPS could mean the number of epochs or mini-batches. This `STEPS` should be used by the trial to control how long it runs. Refer to the example under `examples/trials/mnist-hyperband/` for details.

`eta` means `n/eta` configurations from `n` configurations will survive and rerun using more STEPS.

Here is a concrete example of `R=81` and `eta=3`:

|  | s=4 | s=3 | s=2 | s=1 | s=0 |
|------|-----|-----|-----|-----|-----|
|i     | n r | n r | n r | n r | n r |
|0     |81 1 |27 3 |9 9  |6 27 |5 81 |
|1     |27 3 |9 9  |3 27 |2 81 |     |
|2     |9 9  |3 27 |1 81 |     |     |
|3     |3 27 |1 81 |     |     |     |
|4     |1 81 |     |     |     |     |

`s` means bucket, `n` means the number of configurations that are generated, the corresponding `r` means how many STEPS these configurations run. `i` means round, for example, bucket 4 has 5 rounds, bucket 3 has 4 rounds.

About how to write trial code, please refer to the instructions under `examples/trials/mnist-hyperband/`.

## 4. To be improved
The current implementation of Hyperband can be further improved by supporting simple early stop algorithm, because it is possible that not all the configurations in the top `n/eta` perform good. The unpromising configurations can be stopped early.

In the current implementation, configurations are generated randomly, which follows the design in the [paper][1]. To further improve, configurations could be generated more wisely by leveraging advanced algorithms.

[1]: https://arxiv.org/pdf/1603.06560.pdf
