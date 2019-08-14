# SMAC Tuner

## SMAC

[SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) 基于 Sequential Model-Based Optimization (SMBO). 它利用使用过的结果好的模型（高斯随机过程模型），并将随机森林引入到 SMBO 中，来处理分类参数。 NNI 的 SMAC 通过包装 [SMAC3](https://github.com/automl/SMAC3) 来支持。

Note that SMAC on nni only supports a subset of the types in [search space spec](../Tutorial/SearchSpaceSpec.md), including `choice`, `randint`, `uniform`, `loguniform`, `quniform`.