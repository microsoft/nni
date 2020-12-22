**How to Launch from Python**
=============================

Overview
--------
This is a new way to launch experiments since ``nni v2.0``. Before, you need to configure our experiment in a yaml configuration file and launch the experiment through ``nnictl`` command. Now, you can also configure and run your experiment directly in the python file. If you are familiar with python programming, this will undoubtedly bring you more convenience.

How to Use
----------
After successfully installing ``nni``, you can write your own experimental configuration and run. You can even use python script to launch multiple experiments as you like! The following three steps are necessary.

..

    Step 1 - Initialize a tuner you want to use


.. code-block:: python

    from nni.algorithms.hpo.foo_tuner import FooTuner
    foo_tuner = FooTuner()

Very simple, you have successfully initialized a ``FooTuner`` instance called ``foo_tuner``.

See all real `builtin tuners <../builtin_tuner.rst>`__ supported in NNI.

..

    Step 2 - Initialize a experiment instance and configure it

.. code-block:: python

    experiment = Experiment(tuner=foo_tuner, training_service='local')

Now, you have a ``Experiment`` instance with ``foo_tuner`` you have initialized in the previous step, and this experiment will launch trials on your local machine due to ``training_service='local'``.

See all `training services <../training_services.rst>`__ supported in NNI.

.. code-block:: python

    experiment.config.experiment_name = 'test'

Use the form like ``experiment.config.foo = 'bar'`` to configure your experiment.

See `parameter configuration <TBD>`__ required by different training services.

..

    Step 3 - Just run

.. code-block:: python

    experiment.run(port=8081)

Now, you have successfully launched an NNI experiment. And you can type ``localhost:8081`` in your browser to observe your experiment in real time.

Example
-------
Below is an example for this new launching approach. You can also find this code in ``examples/trials/mnist-tfv2/launch.py``.

.. code-block:: python

    from pathlib import Path
    from nni.experiment import Experiment
    from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

    tuner = HyperoptTuner('tpe')

    search_space = {
        "dropout_rate": { "_type": "uniform", "_value": [0.5, 0.9] },
        "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
        "hidden_size": { "_type": "choice", "_value": [124, 512, 1024] },
        "batch_size": { "_type": "choice", "_value": [16, 32] },
        "learning_rate": { "_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1] }
    }

    experiment = Experiment(tuner, 'local')
    experiment.config.experiment_name = 'test'
    experiment.config.trial_concurrency = 2
    experiment.config.max_trial_number = 5
    experiment.config.search_space = search_space
    experiment.config.trial_command = 'python3 mnist.py'
    experiment.config.trial_code_directory = Path(__file__).parent
    experiment.config.training_service.use_active_gpu = True

    experiment.run(8081)

API
---

..  autoclass:: nni.experiment.Experiment
    :members: