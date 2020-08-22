# Evolution NAS Tuner on NNI

## EvoNasTuner

This is a tuner geared for NNI’s Neural Architecture Search (NAS) interface. It uses the [evolution algorithm](https://arxiv.org/pdf/1802.01548.pdf).

The tuner first randomly initializes the number of `population` models and evaluates them. After that, every time to produce a new architecture, the tuner randomly chooses the number of `sample` architectures from `population`, then mutates the best model in `sample`, the parent model, to produce the child model. The mutation includes the hidden mutation and the op mutation. The hidden state mutation consists of replacing a hidden state with another hidden state from within the cell, subject to the constraint that no loops are formed. The op mutation behaves like the hidden state mutation as far as replacing one op with another op from the op set. Note that keeping the child model the same as its parent is not allowed. After evaluating the child model, it is added to the tail of the `population`, then pops the front one.

The whole procedure is summarized by the pseudocode below.

![img](../../img/EvoNasTuner.png)
