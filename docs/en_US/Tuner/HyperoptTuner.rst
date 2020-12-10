TPE, Random Search, Anneal Tuners on NNI
========================================

TPE
---

The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. The TPE approach models P(x|y) and P(y) where x represents hyperparameters and y the associated evaluation matric. P(x|y) is modeled by transforming the generative process of hyperparameters, replacing the distributions of the configuration prior with non-parametric densities. This optimization approach is described in detail in `Algorithms for Hyper-Parameter Optimization <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__. ​

Parallel TPE optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

TPE approaches were actually run asynchronously in order to make use of multiple compute nodes and to avoid wasting time waiting for trial evaluations to complete. The original algorithm design was optimized for sequential computation. If we were to use TPE with much concurrency, its performance will be bad. We have optimized this case using the Constant Liar algorithm. For these principles of optimization, please refer to our `research blog <../CommunitySharings/ParallelizingTpeSearch.rst>`__.

Usage
^^^^^

 To use TPE, you should add the following spec in your experiment's YAML config file:

.. code-block:: yaml

   tuner:
     builtinTunerName: TPE
     classArgs:
       optimize_mode: maximize
       parallel_optimize: True
       constant_liar_type: min

**classArgs requirements:**


* **optimize_mode** (*maximize or minimize, optional, default = maximize*\ ) - If 'maximize', tuners will try to maximize metrics. If 'minimize', tuner will try to minimize metrics.
* **parallel_optimize** (*bool, optional, default = False*\ ) - If True, TPE will use the Constant Liar algorithm to optimize parallel hyperparameter tuning. Otherwise, TPE will not discriminate between sequential or parallel situations.
* **constant_liar_type** (*min or max or mean, optional, default = min*\ ) - The type of constant liar to use, will logically be determined on the basis of the values taken by y at X. There are three possible values, min{Y}, max{Y}, and mean{Y}.

Random Search
-------------

In `Random Search for Hyper-Parameter Optimization <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`__ we show that Random Search might be surprisingly effective despite its simplicity. We suggest using Random Search as a baseline when no knowledge about the prior distribution of hyper-parameters is available.

Anneal
------

This simple annealing algorithm begins by sampling from the prior but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on random search that leverages smoothness in the response surface. The annealing rate is not adaptive.
