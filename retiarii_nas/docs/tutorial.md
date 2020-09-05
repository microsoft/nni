# Retiarii Tutorial

In this document, we will illustrate how to design and tune your neural model with Retiarii. We refer to designing and tuning a neural model as an experiment.

## Write your model code

The model code is mainly composed of three parts.

1. base model
2. training approach
3. applied mutators and experiment config

The example code is [mnist_search.py](../mnist_search.py)

## Write a new strategy

To write a new strategy, users can use the APIs in execution.py to submit/train/wait generated graphs. Operate graph object to generate new graphs and retrieve metrics/status of an executed graph. Use `sdk.experiment` to obtain experiment information, e.g., base graph, applied mutators, specified sampler.