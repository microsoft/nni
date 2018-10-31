Hyperband on nni
===

## 1.Introduction
[Hyperband][1] is a popular automl algorithm. The basic idea of Hyperband is that it creates several brackets, each bracket has `n` randomly generated hyperparameter configurations, each configuration uses `r` resource (e.g., epoch number, batch number). After the `n` configurations is finished, it chooses top `n/eta` configurations and runs them using increased `r*eta` resource. At last, it chooses the best configuration it has found so far.

## 2.Implementation with fully parallelism
Frist, this is an example of how to write an automl algorithm based on MsgDispatcherBase, rather than Tuner and Assessor. Hyperband is implemented in this way because it integrates the functions of both Tuner and Assessor.

Second, this implementation fully leverages Hyperband's internal parallelism. More specifically, the next bracket is not started strictly after the current bracket, instead, it starts when there is available resource.

## 3.Usage
TBD

[1]: https://arxiv.org/pdf/1603.06560.pdf