How to Launch an Experiment from Python
=======================================

.. code-block::

    ..  toctree::
        :hidden:

        Start Usage <python_api_start>
        Connect Usage <python_api_connect>

Overview
--------

Since ``v2.0``, NNI provides a new way to launch the experiments. Before that, you need to configure the experiment in the YAML configuration file and then use the ``nnictl`` command to launch the experiment. Now, you can also configure and run experiments directly in the Python file. If you are familiar with Python programming, this will undoubtedly bring you more convenience.

Run a New Experiment
--------------------

After successfully installing ``nni`` and prepare the `trial code <../TrialExample/Trials.rst>`__, you can start the experiment with a Python script in the following 2 steps.

Step 1 - Initialize an experiment instance and configure it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

See `configuration reference <../reference/experiment_config.rst>`__ for more detailed usage of these fields.


Step 2 - Just run
^^^^^^^^^^^^^^^^^

.. code-block:: python

    experiment.run(port=8080)

Now, you have successfully launched an NNI experiment. And you can type ``localhost:8080`` in your browser to observe your experiment in real time.

In this way, experiment will run in the foreground and will automatically exit when the experiment finished. 

.. Note:: If you want to run an experiment in an interactive way, use ``start()`` in Step 2. If you launch the experiment in Python script, please use ``run()``, as ``start()`` is designed for the interactive scenarios.

Example
^^^^^^^

Below is an example for this new launching approach. You can find this code in :githublink:`mnist-tfv2/launch.py <examples/trials/mnist-tfv2/launch.py>`.

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

NNI migrates the API in ``NNI Client`` to this new launching approach. Launch the experiment by ``start()`` instead of ``run()``, then you can use these APIs in interactive mode.

Please refer to `example usage <./python_api_start.rst>`__ and code file :githublink:`python_api_start.ipynb <examples/trials/sklearn/classification/python_api_start.ipynb>`.

.. Note:: ``run()`` polls the experiment status and will automatically call ``stop()`` when the experiment finished. ``start()`` just launched a new experiment, so you need to manually stop the experiment by calling ``stop()``.


Connect and Manage an Exist Experiment
--------------------------------------

If you launch an experiment by ``nnictl`` and also want to use these APIs, you can use ``Experiment.connect()`` to connect to an existing experiment.

Please refer to `example usage <./python_api_connect.rst>`__ and code file :githublink:`python_api_connect.ipynb <examples/trials/sklearn/classification/python_api_connect.ipynb>`.

.. Note:: You can use ``stop()`` to stop the experiment when connecting to an existing experiment.

Resume/View and Manage a Stopped Experiment
-------------------------------------------

You can use ``Experiment.resume()`` and ``Experiment.view()`` to resume and view a stopped experiment, these functions behave like ``nnictl resume`` and ``nnictl view``.

If you want to manage the experiment, set ``wait_completion`` as ``False`` and the functions will return an ``Experiment`` instance. For more parameters, please refer to API reference.


API Reference
-------------

Detailed usage could be found `here <../reference/experiment_config.rst>`__. 

* `Experiment`_
* `Experiment Config <#Experiment-Config>`_
* `Algorithm Config <#Algorithm-Config>`_
* `Training Service Config <#Training-Service-Config>`_
  * `Local Config <#Local-Config>`_ 
  * `Remote Config <#Remote-Config>`_
  * `Openpai Config <#Openpai-Config>`_
  * `AML Config <#AML-Config>`_
* `Shared Storage Config <Shared-Storage-Config>`_


Experiment
^^^^^^^^^^

..  autoclass:: nni.experiment.Experiment
    :members:


Experiment Config
^^^^^^^^^^^^^^^^^

..  autoattribute:: nni.experiment.config.ExperimentConfig.experiment_name

..  autoattribute:: nni.experiment.config.ExperimentConfig.search_space_file

..  autoattribute:: nni.experiment.config.ExperimentConfig.search_space

..  autoattribute:: nni.experiment.config.ExperimentConfig.trial_command

..  autoattribute:: nni.experiment.config.ExperimentConfig.trial_code_directory

..  autoattribute:: nni.experiment.config.ExperimentConfig.trial_concurrency

..  autoattribute:: nni.experiment.config.ExperimentConfig.trial_gpu_number

..  autoattribute:: nni.experiment.config.ExperimentConfig.max_experiment_duration

..  autoattribute:: nni.experiment.config.ExperimentConfig.max_trial_number

..  autoattribute:: nni.experiment.config.ExperimentConfig.nni_manager_ip

..  autoattribute:: nni.experiment.config.ExperimentConfig.use_annotation

..  autoattribute:: nni.experiment.config.ExperimentConfig.debug

..  autoattribute:: nni.experiment.config.ExperimentConfig.log_level

..  autoattribute:: nni.experiment.config.ExperimentConfig.experiment_working_directory

..  autoattribute:: nni.experiment.config.ExperimentConfig.tuner_gpu_indices

..  autoattribute:: nni.experiment.config.ExperimentConfig.tuner

..  autoattribute:: nni.experiment.config.ExperimentConfig.assessor

..  autoattribute:: nni.experiment.config.ExperimentConfig.advisor

..  autoattribute:: nni.experiment.config.ExperimentConfig.training_service

..  autoattribute:: nni.experiment.config.ExperimentConfig.shared_storage


Algorithm Config
^^^^^^^^^^^^^^^^

..  autoattribute:: nni.experiment.config.AlgorithmConfig.name

..  autoattribute:: nni.experiment.config.AlgorithmConfig.class_args

..  autoattribute:: nni.experiment.config.CustomAlgorithmConfig.class_name

..  autoattribute:: nni.experiment.config.CustomAlgorithmConfig.code_directory

..  autoattribute:: nni.experiment.config.CustomAlgorithmConfig.class_args


Training Service Config
^^^^^^^^^^^^^^^^^^^^^^^

Local Config
************

..  autoattribute:: nni.experiment.config.LocalConfig.platform

..  autoattribute:: nni.experiment.config.LocalConfig.use_active_gpu

..  autoattribute:: nni.experiment.config.LocalConfig.max_trial_number_per_gpu

..  autoattribute:: nni.experiment.config.LocalConfig.gpu_indices

Remote Config
*************

..  autoattribute:: nni.experiment.config.RemoteConfig.platform

..  autoattribute:: nni.experiment.config.RemoteConfig.reuse_mode

..  autoattribute:: nni.experiment.config.RemoteConfig.machine_list

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.host

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.port

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.user

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.password

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.ssh_key_file

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.ssh_passphrase

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.use_active_gpu

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.max_trial_number_per_gpu

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.gpu_indices

..  autoattribute:: nni.experiment.config.RemoteMachineConfig.python_path


Openpai Config
**************

..  autoattribute:: nni.experiment.config.OpenpaiConfig.platform

..  autoattribute:: nni.experiment.config.OpenpaiConfig.host

..  autoattribute:: nni.experiment.config.OpenpaiConfig.username

..  autoattribute:: nni.experiment.config.OpenpaiConfig.token

..  autoattribute:: nni.experiment.config.OpenpaiConfig.trial_cpu_number

..  autoattribute:: nni.experiment.config.OpenpaiConfig.trial_memory_size

..  autoattribute:: nni.experiment.config.OpenpaiConfig.storage_config_name

..  autoattribute:: nni.experiment.config.OpenpaiConfig.docker_image

..  autoattribute:: nni.experiment.config.OpenpaiConfig.local_storage_mount_point

..  autoattribute:: nni.experiment.config.OpenpaiConfig.container_storage_mount_point

..  autoattribute:: nni.experiment.config.OpenpaiConfig.reuse_mode

..  autoattribute:: nni.experiment.config.OpenpaiConfig.openpai_config

..  autoattribute:: nni.experiment.config.OpenpaiConfig.openpai_config_file

AML Config
**********

..  autoattribute:: nni.experiment.config.AmlConfig.platform

..  autoattribute:: nni.experiment.config.AmlConfig.subscription_id

..  autoattribute:: nni.experiment.config.AmlConfig.resource_group

..  autoattribute:: nni.experiment.config.AmlConfig.workspace_name

..  autoattribute:: nni.experiment.config.AmlConfig.compute_target

..  autoattribute:: nni.experiment.config.AmlConfig.docker_image

..  autoattribute:: nni.experiment.config.AmlConfig.max_trial_number_per_gpu


Shared Storage Config
^^^^^^^^^^^^^^^^^^^^^

Nfs Config
**********

..  autoattribute:: nni.experiment.config.NfsConfig.storage_type

..  autoattribute:: nni.experiment.config.NfsConfig.nfs_server

..  autoattribute:: nni.experiment.config.NfsConfig.exported_directory

Azure Blob Config
*****************

..  autoattribute:: nni.experiment.config.AzureBlobConfig.storage_type

..  autoattribute:: nni.experiment.config.AzureBlobConfig.storage_account_name

..  autoattribute:: nni.experiment.config.AzureBlobConfig.storage_account_key

..  autoattribute:: nni.experiment.config.AzureBlobConfig.container_name
