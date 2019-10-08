Metis Tuner on NNI
===

## Metis Tuner

[Metis](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/) offers the following benefits when it comes to tuning parameters: While most tools only predicts the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guess work!

While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter.

While most tools have problems of being exploitation-heavy, Metis' search strategy balances exploration, exploitation, and (optional) re-sampling.

Metis belongs to the class of sequential model-based optimization (SMBO), and it is based on the Bayesian Optimization framework. To model the parameter-vs-performance space, Metis uses both Gaussian Process and GMM. Since each trial can impose a high time cost, Metis heavily trades inference computations with naive trial. At each iteration, Metis does two tasks:

It finds the global optimal point in the Gaussian Process space. This point represents the optimal configuration.

It identifies the next hyper-parameter candidate. This is achieved by inferring the potential information gain of exploration, exploitation, and re-sampling.

Note that the only acceptable types of search space are `quniform`, `uniform` and `randint` and numerical `choice`.

More details can be found in our paper: https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/