TPE, Random Search, Anneal Tuners on NNI
===

## TPE

The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. The TPE approach models P(x|y) and P(y) where x represents hyperparameters and y the associated evaluate matric. P(x|y) is modeled by transforming the generative process of hyperparameters, replacing the distributions of the configuration prior with non-parametric densities. This optimization approach is described in detail in [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf). ​

### Parallel TPE optimization

TPE approaches were actually run asynchronously in order to make use of multiple compute nodes and to avoid wasting time waiting for trial evaluations to complete. The original intention of the algorithm design is to optimize sequential. When we use TPE with a large concurrency, its performance will be bad. We have optimized this phenomenon using Constant Liar algorithm. For the principle of optimization, please refer to our [research blog](../CommunitySharings/ParallelizingTpeSearch.md).

### Usage

 To use TPE, you should add the following spec in your experiment's YAML config file:

 ```yaml
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
    parallel_optimize: True
    constant_liar_type: min
```

**Requirement of classArg**
* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will target to maximize metrics. If 'minimize', tuner will target to minimize metrics.
* **parallel_optimize** (*bool, optional, default = False*) - If True, TPE will use Constant Liar algorithm to optimize parallel hyperparameter tuning. Otherwise, TPE will not discriminate between sequential or parallel situations.
* **constant_liar_type** (*min or max or mean, optional, default = min*) - The type of constant liar to use, will logically be determined on the basis of the values taken by y at X. Corresponding to three values, min{Y}, max{Y}, and mean{Y}.


## Random Search

In [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) show that Random Search might be surprisingly simple and effective. We suggests that we could use Random Search as baseline when we have no knowledge about the prior distribution of hyper-parameters.

## Anneal

This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on random search that leverages smoothness in the response surface. The annealing rate is not adaptive.
