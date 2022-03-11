Tuner: Tuning Algorithms
========================

The tuner decides which hyperparameter sets will be evaluated. It is a most important part of NNI HPO.

A tuner works in following steps:

 1. Initialize with a search space.
 2. Generate hyperparameter sets from the search space.
 3. Send hyperparameters to trials.
 4. Receive evaluation results.
 5. Update internal states according to the results.
 6. Go to step 2, until experiment end.

NNI has out-of-the-box support for many popular tuning algorithms. 
They should be sufficient to cover most typical machine learning scenarios.

However, if you have a very specific demand, or if you have designed an algorithm yourself,
you can also implement your own tuner: :doc:`custom_algorithm`

Common Usage
------------

All built-in tuners have similar usage.

To use a built-in tuner, you need to specify its name and arguments in experiment config,
and provides a standard :doc:`search_space`.
Some tuners, like SMAC and DNGO, have extra dependencies that need to be installed separately.

Please check each tuner's reference page for what arguments it supports and whether it needs extra dependencies.

For a general example, random tuner can be configured as follow:

.. code-block:: python

    config.search_space = {
        'x': {'_type': 'uniform', '_value': [0, 1]}
    }
    config.tuner.name = 'Random'
    config.tuner.class_args = {'seed': 0}

Full List
---------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Tuner
     - Brief Introduction of Algorithm

   * - `TPE <../autotune_ref.html#nni.algorithms.hpo.tpe_tuner.TpeTuner>`_
     - The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. `Reference Paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__

   * - `Random Search <../autotune_ref.html#nni.algorithms.hpo.random_tuner.RandomTuner>`_
     - In Random Search for Hyper-Parameter Optimization show that Random Search might be surprisingly simple and effective. We suggest that we could use Random Search as the baseline when we have no knowledge about the prior distribution of hyper-parameters. `Reference Paper <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`__

   * - `Anneal <../autotune_ref.html#nni.algorithms.hpo.hyperopt_tuner.HyperoptTuner>`_
     - This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on the random search that leverages smoothness in the response surface. The annealing rate is not adaptive.

   * - `Naive Evolution <../autotune_ref.html#nni.algorithms.hpo.evolution_tuner.EvolutionTuner>`_
     - Naive Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population-based on search space. For each generation, it chooses better ones and does some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naïve Evolution requires many trials to work, but it's very simple and easy to expand new features. `Reference paper <https://arxiv.org/pdf/1703.01041.pdf>`__

   * - `SMAC <../autotune_ref.html#nni.algorithms.hpo.smac_tuner.SMACTuner>`_
     - SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by NNI is a wrapper on the SMAC3 GitHub repo.

       Notice, SMAC needs to be installed by ``pip install nni[SMAC]`` command. `Reference Paper, <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ `GitHub Repo <https://github.com/automl/SMAC3>`__

   * - `Batch <../autotune_ref.html#nni.algorithms.hpo.batch_tuner.BatchTuner>`_
     - Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in search space spec.

   * - `Grid Search <../autotune_ref.html#nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner>`_
     - Grid Search performs an exhaustive searching through the search space.

   * - `Hyperband <../autotune_ref.html#nni.algorithms.hpo.hyperband_advisor.Hyperband>`_
     - Hyperband tries to use limited resources to explore as many configurations as possible and returns the most promising ones as a final result. The basic idea is to generate many configurations and run them for a small number of trials. The half least-promising configurations are thrown out, the remaining are further trained along with a selection of new configurations. The size of these populations is sensitive to resource constraints (e.g. allotted search time). `Reference Paper <https://arxiv.org/pdf/1603.06560.pdf>`__

   * - `Metis <../autotune_ref.html#nni.algorithms.hpo.metis_tuner.MetisTuner>`_
     - Metis offers the following benefits when it comes to tuning parameters: While most tools only predict the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guesswork. While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter. `Reference Paper <https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/>`__

   * - `BOHB <../autotune_ref.html#nni.algorithms.hpo.bohb_advisor.BOHB>`_
     - BOHB is a follow-up work to Hyperband. It targets the weakness of Hyperband that new configurations are generated randomly without leveraging finished trials. For the name BOHB, HB means Hyperband, BO means Bayesian Optimization. BOHB leverages finished trials by building multiple TPE models, a proportion of new configurations are generated through these models. `Reference Paper <https://arxiv.org/abs/1807.01774>`__

   * - `GP <../autotune_ref.html#nni.algorithms.hpo.gp_tuner.GPTuner>`_
     - Gaussian Process Tuner is a sequential model-based optimization (SMBO) approach with Gaussian Process as the surrogate. `Reference Paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__, `Github Repo <https://github.com/fmfn/BayesianOptimization>`__

   * - `PBT <../autotune_ref.html>`_
     - PBT Tuner is a simple asynchronous optimization algorithm which effectively utilizes a fixed computational budget to jointly optimize a population of models and their hyperparameters to maximize performance. `Reference Paper <https://arxiv.org/abs/1711.09846v1>`__

   * - `DNGO <../autotune_ref.html>`_
     - Use of neural networks as an alternative to GPs to model distributions over functions in bayesian optimization.

Comparison
----------

These articles have compared built-in tuners' performance on some different tasks:

:doc:`hpo_benchmark_stats`

:doc:`/CommunitySharings/HpoComparison`
