TPE Tuner
=========

Introduction
------------

The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach.
SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements,
and then subsequently choose new hyperparameters to test based on this model.

The TPE approach models P(x|y) and P(y) where x represents hyperparameters and y the associated evaluation matric.
P(x|y) is modeled by transforming the generative process of hyperparameters,
replacing the distributions of the configuration prior with non-parametric densities.

This optimization approach is described in detail in `Algorithms for Hyper-Parameter Optimization <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__.

Parallel TPE optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

TPE approaches were actually run asynchronously in order to make use of multiple compute nodes and to avoid wasting time waiting for trial evaluations to complete.
The original algorithm design was optimized for sequential computation.
If we were to use TPE with much concurrency, its performance will be bad.
We have optimized this case using the Constant Liar algorithm.
For these principles of optimization, please refer to our `research blog <../CommunitySharings/ParallelizingTpeSearch.rst>`__.

Usage
-----

 To use TPE, you should add the following spec in your experiment's YAML config file:

.. code-block:: yaml

    ## minimal config ##
    tuner:
      name: TPE
      classArgs:
        optimize_mode: minimize

.. code-block:: yaml

    ## advanced config ##
    tuner:
      name: TPE
      classArgs:
        optimize_mode: maximize
        seed: 12345
        tpe_args:
          constant_liar_type: 'mean'
          n_startup_jobs: 10
          n_ei_candidates: 20
          linear_forgetting: 100
          prior_weight: 0
          gamma: 0.5
        
classArgs
^^^^^^^^^

.. list-table::
    :widths: 10 20 10 60
    :header-rows: 1

    * - Field
      - Type
      - Default
      - Description
    
    * - ``optimize_mode``
      - ``'minimize' | 'maximize'``
      - ``'minimize'``
      - Whether to minimize or maximize trial metrics.

    * - ``seed``
      - ``int | null``
      - ``null``
      - The random seed.

    * - ``tpe_args.constant_liar_type``
      - ``'best' | 'worst' | 'mean' | null``
      - ``'best'``
      - TPE algorithm itself does not support parallel tuning. This parameter specifies how to optimize for trial_concurrency > 1. How each liar works is explained in paper's section 6.1.

        In general ``best`` suit for small trial number and ``worst`` suit for large trial number.

    * - ``tpe_args.n_startup_jobs``
      - ``int``
      - ``20``
      - The first N hyper-parameters are generated fully randomly for warming up.

        If the search space is large, you can increase this value. Or if max_trial_number is small, you may want to decrease it.

    * - ``tpe_args.n_ei_candidates``
      - ``int``
      - ``24``
      - For each iteration TPE samples EI for N sets of parameters and choose the best one. (loosely speaking)

    * - ``tpe_args.linear_forgetting``
      - ``int``
      - ``25``
      - TPE will lower the weights of old trials. This controls how many iterations it takes for a trial to start decay.

    * - ``tpe_args.prior_weight``
      - ``float``
      - ``1.0``
      - TPE treats user provided search space as prior.
        When generating new trials, it also incorporates the prior in trial history by transforming the search space to
        one trial configuration (i.e., each parameter of this configuration chooses the mean of its candidate range).
        Here, prior_weight determines the weight of this trial configuration in the history trial configurations.

        With prior weight 1.0, the search space is treated as one good trial.
        For example, "normal(0, 1)" effectly equals to a trial with x = 0 which has yielded good result.

    * - ``tpe_args.gamma``
      - ``float``
      - ``0.25``
      - Controls how many trials are considered "good".

        The number is calculated as "min(gamma * sqrt(N), linear_forgetting)".
