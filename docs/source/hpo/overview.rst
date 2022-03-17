Hyperparameter Optimization Overview
====================================

Auto hyperparameter optimization (HPO), or auto tuning, is one of the key features of NNI.

Introduction to HPO
-------------------

In machine learning, a hyperparameter is a parameter whose value is used to control learning process [1]_,
and HPO is the problem of choosing a set of optimal hyperparameters for a learning algorithm [2]_.

.. [1] https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)
.. [2] https://en.wikipedia.org/wiki/Hyperparameter_optimization

Following code snippet demonstrates a naive HPO process:

.. code-block:: python

    best_hyperparameters = None
    best_accuracy = 0

    for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
        for momentum in [i / 10 for i in range(10)]:
            for activation_type in ['relu', 'tanh', 'sigmoid']:
                model = build_model(activation_type)
                train_model(model, learning_rate, momentum)
                accuracy = evaluate_model(model)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyperparameters = (learning_rate, momentum, activation_type)

     print('Best hyperparameters:', best_hyperparameters)

You may have noticed, the example will train 4×10×3=120 models in total.
Since it consumes so much computing resources, you may want to:

 1. Find the best set of hyperparameters with less iterations.
 2. Train the models on distributed platforms.
 3. Have a portal to monitor and control the process.

And NNI will do them for you.

Key Features of NNI HPO
-----------------------

Tuning Algorithms
^^^^^^^^^^^^^^^^^

NNI provides *tuners* to speed up the process of finding best hyperparameter set.

A tuner, or a tuning algorithm, decides the order in which hyperparameter sets are evaluated.
Based on the results of historical hyperparameter sets, an efficient tuner can predict where the best hyperparameters locates around,
and finds them in much fewer attempts.

The naive example above evaluates all possible hyperparameter sets in constant order, ignoring the historical results.
This is the brute-force tuning algorithm called *grid search*.

NNI has out-of-the-box support for a variety of popular tuners.
It includes naive algorithms like random search and grid search, Bayesian-based algorithms like TPE and SMAC,
RL based algorithms like PPO, and much more.

Main article: :doc:`tuners`

Training Platforms
^^^^^^^^^^^^^^^^^^

If you are not interested in distributed platforms, you can simply run NNI HPO with current computer,
just like any ordinary Python library.

And when you want to leverage more computing resources, NNI provides built-in integration for training platforms
from simple on-premise servers to scalable commercial clouds.

With NNI you can write one piece of model code, and concurrently evaluate hyperparameter sets on local machine, SSH servers,
Kubernetes-based clusters, AzureML service, and much more.

Main article: (FIXME: link to training_services)

Web UI
^^^^^^

NNI provides a web portal to monitor training progress, to visualize hyperparameter performance,
to manually customize hyperparameters, and to manage multiple HPO experiments.

(FIXME: image and link)

Tutorials
---------

To start using NNI HPO, choose the tutorial of your favorite framework:

* PyTorch MNIST tutorial
* :doc:`TensorFlow MNIST tutorial </tutorials/hpo_quickstart_tensorflow/main>`

Extra Features
--------------

After you are familiar with basic usage, you can explore more HPO features:

* :doc:`Assessor: Early stop non-optimal models <assessors>`
* :doc:`nnictl: Use command line tool to create and manage experiments </reference/nnictl>`
* :doc:`Custom tuner: Implement your own tuner <custom_algorithm>`
* :doc:`Tensorboard support <tensorboard>`
* :doc:`Tuner benchmark <hpo_benchmark>`
* :doc:`NNI Annotation (legacy) <nni_annotation>`

Built-in Algorithms
-------------------

Tuning Algorithms
^^^^^^^^^^^^^^^^^

Main article: :doc:`tuners`

.. list-table::
    :header-rows: 1
    :widths: auto

    * - Name
      - Category
      - Brief Description

    * - :class:`Random <nni.algorithms.hpo.random_tuner.RandomTuner>`
      - Basic
      - Naive random search.

    * - :class:`GridSearch <nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner>`
      - Basic
      - Brute-force search.

    * - :class:`TPE <nni.algorithms.hpo.tpe_tuner.TpeTuner>`
      - Bayesian
      - Tree-structured Parzen Estimator.

    * - :class:`Anneal <nni.algorithms.hpo.hyperopt_tuner.HyperoptTuner>`
      - Classic
      - Simulated annealing algorithm.

    * - :class:`Evolution <nni.algorithms.hpo.evolution_tuner.EvolutionTuner>`
      - Classic
      - Naive evolution algorithm.

    * - :class:`SMAC <nni.algorithms.hpo.smac_tuner.SMACTuner>`
      - Bayesian
      - Sequential Model-based optimization for general Algorithm Configuration.

    * - :class:`Hyperband <nni.algorithms.hpo.hyperband_advisor.Hyperband>`
      - Advanced
      - Evaluate more hyperparameter sets by adaptively allocating resources.

    * - :class:`MetisTuner <nni.algorithms.hpo.metis_tuner.MetisTuner>`
      - Bayesian
      - Robustly optimizing tail latencies of cloud systems.

    * - :class:`BOHB <nni.algorithms.hpo.bohb_advisor.BOHB>`
      - Advanced
      - Bayesian Optimization with HyperBand.

    * - :class:`GPTuner <nni.algorithms.hpo.gp_tuner.GPTuner>`
      - Bayesian
      - Gaussian Process.

    * - :class:`PBTTuner <nni.algorithms.hpo.pbt_tuner.PBTTuner>`
      - Advanced
      - Population Based Training of neural networks.

    * - :class:`DNGOTuner <nni.algorithms.hpo.dngo_tuner.DNGOTuner>`
      - Bayesian
      - (FIXME: full name?)

    * - :class:`PPOTuner <nni.algorithms.hpo.ppo_tuner.PPOTuner>`
      - RL
      - Proximal Policy Optimization.

    * - :class:`BatchTuner <nni.algorithms.hpo.batch_tuner.BatchTuner>`
      - Basic
      - Manually specify hyperparameter sets.

Early Stopping
^^^^^^^^^^^^^^

Main article: :doc:`assessors`

.. list-table::
    :header-rows: 1
    :widths: auto

    * - Name
      - Brief Description

    * - :class:`Medianstop <nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor>`
      - Stop if the hyperparameter set performs worse than median at any step.

    * - :class:`Curvefitting <nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor>`
      - Stop if the learning curve will likely converge to suboptimal result.
