**How to Launch an Experiment from Python**
===========================================

..  toctree::
    :hidden:

    Start Usage <python_api_start>
    Connect Usage <python_api_connect>

Overview
--------
Since ``nni v2.0``, we provide a new way to launch experiments. Before that, you need to configure the experiment in the yaml configuration file and then use the experiment ``nnictl`` command to launch the experiment. Now, you can also configure and run experiments directly in python file. If you are familiar with python programming, this will undoubtedly bring you more convenience.

Run a New Experiment
--------------------
After successfully installing ``nni``, you can start the experiment with a python script in the following 2 steps.

..

    Step 1 - Initialize an experiment instance and configure it

.. code-block:: python

    from nni.experiment import Experiment
    experiment = Experiment('local')

Now, you have a ``Experiment`` instance, and this experiment will launch trials on your local machine due to ``training_service='local'``.

See all `training services <../training_services.rst>`__ supported in NNI.

.. code-block:: python

    experiment.config.experiment_name = 'MNIST example'
    experiment.config.trial_concurrency = 2
    experiment.config.max_trial_number = 10
    experiment.config.search_space = search_space
    experiment.config.trial_command = 'python3 mnist.py'
    experiment.config.trial_code_directory = Path(__file__).parent
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.training_service.use_active_gpu = True

Use the form like ``experiment.config.foo = 'bar'`` to configure your experiment.

See all real `builtin tuners <../builtin_tuner.rst>`__ supported in NNI.

See `parameter configuration <../reference/experiment_config.rst>`__ required by different training services.

..

    Step 2 - Just run

.. code-block:: python

    experiment.run(port=8080)

Now, you have successfully launched an NNI experiment. And you can type ``localhost:8080`` in your browser to observe your experiment in real time.

.. Note:: In this way, experiment will run in the foreground and will automatically exit when the experiment finished. If you want to run an experiment in an interactive way, use ``start()`` in Step 2. 

Example
^^^^^^^
Below is an example for this new launching approach. You can also find this code in :githublink:`mnist-tfv2/launch.py <examples/trials/mnist-tfv2/launch.py>`.

.. code-block:: python

    from pathlib import Path

    from nni.experiment import Experiment

    search_space = {
        "dropout_rate": { "_type": "uniform", "_value": [0.5, 0.9] },
        "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
        "hidden_size": { "_type": "choice", "_value": [124, 512, 1024] },
        "batch_size": { "_type": "choice", "_value": [16, 32] },
        "learning_rate": { "_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1] }
    }

    experiment = Experiment('local')
    experiment.config.experiment_name = 'MNIST example'
    experiment.config.trial_concurrency = 2
    experiment.config.max_trial_number = 10
    experiment.config.search_space = search_space
    experiment.config.trial_command = 'python3 mnist.py'
    experiment.config.trial_code_directory = Path(__file__).parent
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.training_service.use_active_gpu = True

    experiment.run(8080)

Start and Manage a New Experiment
---------------------------------
We migrate the API in ``NNI Client`` to this new launching approach.
Launch the experiment by ``start()`` instead of ``run()``, then you can use these APIs in interactive mode.

Please refer to `example usage <./python_api_start.rst>`__ and code file :githublink:`python_api_start.ipynb <examples/trials/sklearn/classification/python_api_start.ipynb>`.

.. Note:: ``run()`` polls the experiment status and will automatically call ``stop()`` when the experiment finished. ``start()`` just launched a new experiment, so you need to manually stop the experiment by calling ``stop()``.

Connect and Manage an Exist Experiment
--------------------------------------
If you launch the experiment by ``nnictl`` and also want to use these APIs, you can use ``Experiment.connect()`` to connect to an existing experiment.

Please refer to `example usage <./python_api_connect.rst>`__ and code file :githublink:`python_api_connect.ipynb <examples/trials/sklearn/classification/python_api_connect.ipynb>`.

.. Note:: You can use ``stop()`` to stop the experiment when connecting to an existing experiment.

Resume/View and Manage a Stopped Experiment
-------------------------------------------

You can use ``Experiment.resume()`` and ``Experiment.view()`` to resume and view a stopped experiment, these functions behave like ``nnictl resume`` and ``nnictl view``.
If you want to manage the experiment, set ``wait_completion`` as ``False`` and the functions will return an ``Experiment`` instance. For more parameters, please refer to API.

API
---

..  autoclass:: nni.experiment.Experiment
    :members:
