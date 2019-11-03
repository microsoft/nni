GP Tuner on NNI
===

## GP Tuner

Bayesian optimization works by constructing a posterior distribution of functions (Gaussian Process here) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not.

GP Tuner is designed to minimize/maximize the number of steps required to find a combination of parameters that are close to the optimal combination. To do so, this method uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) and common tools can be employed. Therefore Bayesian Optimization is most adequate for situations where sampling the function to be optimized is a very expensive endeavor.

Note that the only acceptable types of search space are `randint`, `uniform`, `quniform`,  `loguniform`, `qloguniform`, and numerical `choice`.

This optimization approach is described in Section 3 of [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf). 
