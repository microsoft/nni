SMAC Tuner on NNI
===

## SMAC

[SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO in order to handle categorical parameters. The SMAC supported by nni is a wrapper on [the SMAC3 github repo](https://github.com/automl/SMAC3).

Note that SMAC on nni only supports a subset of the types in the [search space spec](../Tutorial/SearchSpaceSpec.md): `choice`, `randint`, `uniform`, `loguniform`, and `quniform`.