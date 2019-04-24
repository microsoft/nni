Grid Search on NNI
===

## Grid Search

Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file. Note that the only acceptable types of search space are `choice`, `quniform`, `qloguniform`. **The number `q` in `quniform` and `qloguniform` has special meaning (different from the spec in [search space spec](SearchSpaceSpec.md)). It means the number of values that will be sampled evenly from the range `low` and `high`.**