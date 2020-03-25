PBT Tuners on NNI
===

## PBTTuner

Population Based Training(PBT) comes from [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846v1). It's a simple asynchronous optimization algorithm which effectively utilizes a fixed computational budget to jointly optimize a population of models and their hyperparameters to maximize performance. Importantly, PBT discovers a schedule of hyperparameter settings rather than following the generally sub-optimal strategy of trying to find a single fixed set to use for the whole course of training.