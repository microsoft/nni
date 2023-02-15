Tuner: Tuning Algorithms
========================

The tuner decides which hyperparameter sets will be evaluated. It is a most important part of NNI HPO.

A tuner works like following pseudocode:

.. code-block:: python

    space = get_search_space()
    history = []
    while not experiment_end:
        hp = suggest_hyperparameter_set(space, history)
        result = run_trial(hp)
        history.append((hp, result))

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

As a general example, random tuner can be configured as follow:

.. code-block:: python

    config.search_space = {
        'x': {'_type': 'uniform', '_value': [0, 1]},
        'y': {'_type': 'choice', '_value': ['a', 'b', 'c']}
    }
    config.tuner.name = 'random'
    config.tuner.class_args = {'seed': 0}

Built-in Tuners
---------------

.. list-table::
    :header-rows: 1
    :widths: auto

    * - Tuner
      - Category
      - Brief Introduction

    * - :class:`TPE <nni.algorithms.hpo.tpe_tuner.TpeTuner>`
      - Bayesian
      - Tree-structured Parzen Estimator, a classic Bayesian optimization algorithm.
        (`paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__)

        TPE is a lightweight tuner that has no extra dependency and supports all search space types.
        Good to start with.

        The drawback is that TPE cannot discover relationship between different hyperparameters.

    * - :class:`Random <nni.algorithms.hpo.random_tuner.RandomTuner>`
      - Basic
      - Naive random search, the baseline. It supports all search space types.

    * - :class:`Grid Search <nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner>`
      - Basic
      - Divides search space into evenly spaced grid, and performs brute-force traverse. Another baseline.

        It supports all search space types.
        Recommended when the search space is small, and when you want to find the strictly optimal hyperparameters.

    * - :class:`Anneal <nni.algorithms.hpo.hyperopt_tuner.HyperoptTuner>`
      - Heuristic
      - This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on the random search that leverages smoothness in the response surface. The annealing rate is not adaptive.

        Notice, Anneal needs to be installed by ``pip install nni[Anneal]`` command.

    * - :class:`Evolution <nni.algorithms.hpo.evolution_tuner.EvolutionTuner>`
      - Heuristic
      - Naive Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population-based on search space. For each generation, it chooses better ones and does some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naïve Evolution requires many trials to work, but it's very simple and easy to expand new features. `Reference paper <https://arxiv.org/pdf/1703.01041.pdf>`__

    * - :class:`SMAC <nni.algorithms.hpo.smac_tuner.SMACTuner>`
      - Bayesian
      - SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by NNI is a wrapper on the SMAC3 GitHub repo.

        Notice, SMAC needs to be installed by ``pip install nni[SMAC]`` command. `Reference Paper, <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ `GitHub Repo <https://github.com/automl/SMAC3>`__

    * - :class:`Batch <nni.algorithms.hpo.batch_tuner.BatchTuner>`
      - Basic
      - Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in search space spec.

    * - :class:`Hyperband <nni.algorithms.hpo.hyperband_advisor.Hyperband>`
      - Heuristic
      - Hyperband tries to use limited resources to explore as many configurations as possible and returns the most promising ones as a final result. The basic idea is to generate many configurations and run them for a small number of trials. The half least-promising configurations are thrown out, the remaining are further trained along with a selection of new configurations. The size of these populations is sensitive to resource constraints (e.g. allotted search time). `Reference Paper <https://arxiv.org/pdf/1603.06560.pdf>`__

    * - :class:`Metis <nni.algorithms.hpo.metis_tuner.MetisTuner>`
      - Bayesian
      - Metis offers the following benefits when it comes to tuning parameters: While most tools only predict the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guesswork. While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter. `Reference Paper <https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/>`__

    * - :class:`BOHB <nni.algorithms.hpo.bohb_advisor.BOHB>`
      - Bayesian
      - BOHB is a follow-up work to Hyperband. It targets the weakness of Hyperband that new configurations are generated randomly without leveraging finished trials. For the name BOHB, HB means Hyperband, BO means Bayesian Optimization. BOHB leverages finished trials by building multiple TPE models, a proportion of new configurations are generated through these models. `Reference Paper <https://arxiv.org/abs/1807.01774>`__

    * - :class:`GP <nni.algorithms.hpo.gp_tuner.GPTuner>`
      - Bayesian
      - Gaussian Process Tuner is a sequential model-based optimization (SMBO) approach with Gaussian Process as the surrogate. `Reference Paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__, `Github Repo <https://github.com/fmfn/BayesianOptimization>`__

    * - :class:`PBT <nni.algorithms.hpo.pbt_tuner.PBTTuner>`
      - Heuristic
      - PBT Tuner is a simple asynchronous optimization algorithm which effectively utilizes a fixed computational budget to jointly optimize a population of models and their hyperparameters to maximize performance. `Reference Paper <https://arxiv.org/abs/1711.09846v1>`__

    * - :class:`DNGO <nni.algorithms.hpo.dngo_tuner.DNGOTuner>`
      - Bayesian
      - Use of neural networks as an alternative to GPs to model distributions over functions in bayesian optimization.

Comparison
----------

These articles have compared built-in tuners' performance on some different tasks:

:doc:`hpo_benchmark_stats`

:doc:`/sharings/hpo_comparison`
