Grid Search on nni
===

## 1. Introduction

Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file.

Note that the only acceptable types of search space are `choice`, `quniform`, `qloguniform` since only these three types of search space can be exhausted.

Moreover, in GridSearch Tuner, for users' convenience, the definition of `quniform` and `qloguniform` change, where q here specifies the number of values that will be sampled. Details about them are listed as follows:

* Type 'quniform' will receive three values [low, high, q], where [low, high] specifies a range and 'q' specifies the number of values that will be sampled evenly. Note that q should be at least 2. It will be sampled in a way that the first sampled value is 'low', and each of the following values is (high-low)/q larger that the value in front of it.
* Type 'qloguniform' behaves like 'quniform' except that it will first change the range to [log(low), log(high)] and sample and then change the sampled value back.

## 2. Usage

Since Grid Search Tuner will exhaust all possible hyper-parameter combination according to the search space file without any hyper-parameter for tuner itself, all you need to do is to specify tuner name in your experiment's YAML config file:

```
tuner:
  builtinTunerName: GridSearch
```

