Hyperparameter Optimization Overview
====================================

Auto hyperparameter optimization (HPO), or auto tuning, is one of the key features of NNI.

In machine learning, a hyperparameter is a parameter whose value is used to control learning process [1]_,
and HPO is the problem of choosing a set of optimal hyperparameters for a learning algorithm [2]_.

.. [1] https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)
.. [2] https://en.wikipedia.org/wiki/Hyperparameter_optimization

Following code snippet demonstrates a naive HPO process:

.. code-block:: python

    best_hyperparameters = None
    best_loss = math.inf

    for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
        for momentum in [i / 10 for i in range(10)]:
            for activation_type in ['relu', 'tanh', 'sigmoid']:

                model = build_my_model(activation_type)
                train_my_model(model, learning_rate, momentum)
                loss = evaluate_my_model(model)

                if loss < best_loss:
                    best_loss = loss
                    best_hyperparameters = {
                        'learning_rate': learning_rate,
                        'momentum': momentum,
                        'activation_type': activation_type
                    }

     print('Best hyperparameters:', best_hyperparameters)

The example will train 120 models in total.
Since it consumes so much computing resources, you may want to:

 1. Find the best set of hyperparameters with less iterations.
 2. Train the models on distributed platforms.
 3. Have a portal to monitor and control the process.

NNI will do them for you.

Key Features
------------

Tuning Algorithms
^^^^^^^^^^^^^^^^^

NNI provides *tuners* to speed up the process of finding best hyperparameter set.

A tuner, or a tuning algorithm, decides the order in which hyperparameter sets are evaluated.
Based on the results of historical hyperparameter sets, an efficient tuner can predict where the best set locates around,
and finds it in much fewer attempts.

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

With NNI you can write one piece of model code, and concurrently evaluate its hyperparameter sets on local machine, SSH servers,
Kubernetes-based clusters, AzureML service, and much more.

Main article: (FIXME: link to training_services)

Web UI
^^^^^^

NNI provides a web portal to monitor training progress, visualize hyperparameter performance,
manually customize hyperparameters, and manage multiple HPO experiments.

(FIXME: image and link)

Tutorials
---------

To start using NNI HPO, choose the tutorial of your favorite framework:

  * PyTorch MNIST tutorial
  * TensorFlow MNIST tutorial
  * Scikit-learn classification tutorial

(FIXME: link)

Extra Features
--------------

After you are familiar with the basic usage, you can explore more HPO features:

  * Assessor: Early stop non-optimal models
  * nnictl: Use command line tool to create and manage experiments
  * Custom tuner: Implement your own tuner
  * Tensorboard support
  * Shared storage (experimental)
  * NNI Annotation (legacy)

(FIXME: link)
