Run HPO Experiment with nnictl
==============================

This tutorial has exactly the same effect as :doc:`../hpo_quickstart_pytorch/main`.

Both tutorials optimize the model in `official PyTorch quickstart
<https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__ with auto-tuning,
while this one manages the experiment with command line tool and YAML config file, instead of pure Python code.

The tutorial consists of 4 steps: 

1. Modify the model for auto-tuning.
2. Define hyperparameters' search space.
3. Create config file.
4. Run the experiment.

The first two steps are identical to quickstart.

Step 1: Prepare the model
-------------------------
In first step, we need to prepare the model to be tuned.

The model should be put in a separate script.
It will be evaluated many times concurrently,
and possibly will be trained on distributed platforms.

In this tutorial, the model is defined in :doc:`model.py <model>`.

In short, it is a PyTorch model with 3 additional API calls:

1. Use :func:`nni.get_next_parameter` to fetch the hyperparameters to be evalutated.
2. Use :func:`nni.report_intermediate_result` to report per-epoch accuracy metrics.
3. Use :func:`nni.report_final_result` to report final accuracy.

Please understand the model code before continue to next step.

Step 2: Define search space
---------------------------
In model code, we have prepared 3 hyperparameters to be tuned:
*features*, *lr*, and *momentum*.

Here we need to define their *search space* so the tuning algorithm can sample them in desired range.

Assuming we have following prior knowledge for these hyperparameters:

1. *features* should be one of 128, 256, 512, 1024.
2. *lr* should be a float between 0.0001 and 0.1, and it follows exponential distribution.
3. *momentum* should be a float between 0 and 1.

In NNI, the space of *features* is called ``choice``;
the space of *lr* is called ``loguniform``;
and the space of *momentum* is called ``uniform``.
You may have noticed, these names are derived from ``numpy.random``.

For full specification of search space, check :doc:`the reference </hpo/search_space>`.

Now we can define the search space as follow:

.. code-block:: yaml

    search_space:
      features:
        _type: choice
        _value: [ 128, 256, 512, 1024 ]
      lr:
        _type: loguniform
        _value: [ 0.0001, 0.1 ]
      momentum:
        _type: uniform
        _value: [ 0, 1 ]

Step 3: Configure the experiment
--------------------------------
NNI uses an *experiment* to manage the HPO process.
The *experiment config* defines how to train the models and how to explore the search space.

In this tutorial we use a YAML file ``config.yaml`` to define the experiment.

Configure trial code
^^^^^^^^^^^^^^^^^^^^
In NNI evaluation of each hyperparameter set is called a *trial*.
So the model script is called *trial code*.

.. code-block:: yaml

    trial_command: python model.py
    trial_code_directory: .

When ``trial_code_directory`` is a relative path, it relates to the config file.
So in this case we need to put ``config.yaml`` and ``model.py`` in the same directory.

.. attention::

    The rules for resolving relative path are different in YAML config file and :doc:`Python experiment API </reference/experiment>`.
    In Python experiment API relative paths are relative to current working directory.

Configure how many trials to run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here we evaluate 10 sets of hyperparameters in total, and concurrently evaluate 2 sets at a time.

.. code-block:: yaml

    max_trial_number: 10
    trial_concurrency: 2

You may also set ``max_experiment_duration = '1h'`` to limit running time.

If neither ``max_trial_number`` nor ``max_experiment_duration`` are set,
the experiment will run forever until you stop it.

.. note::

    ``max_trial_number`` is set to 10 here for a fast example.
    In real world it should be set to a larger number.
    With default config TPE tuner requires 20 trials to warm up.


Configure tuning algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^
Here we use :doc:`TPE tuner </hpo/tuners>`.

.. code-block:: yaml

    name: TPE
    class_args:
      optimize_mode: maximize

Configure training service
^^^^^^^^^^^^^^^^^^^^^^^^^^

In this tutorial we use *local* mode,
which means models will be trained on local machine, without using any special training platform.

.. code-block:: yaml

    training_service:
      platform: local

Wrap up
^^^^^^^

The full content of ``config.yaml`` is as follow:

.. code-block:: yaml

    search_space:
      features:
        _type: choice
        _value: [ 128, 256, 512, 1024 ]
      lr:
        _type: loguniform
        _value: [ 0.0001, 0.1 ]
      momentum:
        _type: uniform
        _value: [ 0, 1 ]
    
    trial_command: python model.py
    trial_code_directory: .

    trial_concurrency: 2
    max_trial_number: 10
    
    tuner:
      name: TPE
      class_args:
        optimize_mode: maximize
    
    training_service:
      platform: local

Step 4: Run the experiment
--------------------------
Now the experiment is ready. Launch it with ``nnictl create`` command:

.. code-block:: bash

    $ nnictl create --config config.yaml --port 8080

You can use the web portal to view experiment status: http://localhost:8080.

.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [2022-04-01 12:00:00] Creating experiment, Experiment ID: p43ny6ew
    [2022-04-01 12:00:00] Starting web server...
    [2022-04-01 12:00:01] Setting up...
    [2022-04-01 12:00:01] Web portal URLs: http://127.0.0.1:8080 http://192.168.1.1:8080
    [2022-04-01 12:00:01] To stop experiment run "nnictl stop p43ny6ew" or "nnictl stop --all"
    [2022-04-01 12:00:01] Reference: https://nni.readthedocs.io/en/stable/reference/nnictl.html

When the experiment is done, use ``nnictl stop`` command to stop it.

.. code-block:: bash

    $ nnictl stop p43ny6ew

.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    INFO:  Stopping experiment 7u8yg9zw
    INFO:  Stop experiment success.
