# SMAC Tuner

## SMAC

[SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) 基于 Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO in order to handle categorical parameters. NNI 的 SMAC 通过包装 [SMAC3](https://github.com/automl/SMAC3) 来支持。

Note that SMAC on nni only supports a subset of the types in the [search space spec](../Tutorial/SearchSpaceSpec.md): `choice`, `randint`, `uniform`, `loguniform`, and `quniform`.