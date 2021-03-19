如何从 Python 发起实验
===========================================

..  toctree::
    :hidden:

    启动用法 <python_api_start>
    连接用法 <python_api_connect>

概述
--------
从 ``nni v2.0`` 起，我们提供了一种全新方式发起 Experiment 。 在此之前，您需要在 yaml 文件中配置实验，然后使用 ``nnictl`` 命令启动 Experiment 。 现在，您还可以直接在python文件中配置和运行 Experiment 。 如果您熟悉 Python 编程，那么无疑会为您带来很多便利。

运行一个新的 Experiment
----------------------------------------
After successfully installing ``nni``, you can start the experiment with a python script in the following 2 steps.

..

    Step 1 - Initialize an experiment instance and configure it

.. code-block:: python

    from nni.experiment import Experiment
    experiment = Experiment('local')

Now, you have a ``Experiment`` instance, and this experiment will launch trials on your local machine due to ``training_service='local'``.

查看 NNI 支持的所有 `训练平台 <../training_services.rst>`__。

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

使用类似 ``experiment.config.foo ='bar'`` 的形式来配置您的 Experiment 。

查看 NNI 所有的 `内置 Tuner <../builtin_tuner.rst>`__。

参阅不同平台所需的 `参数配置 <../reference/experiment_config.rst>`__。

..

    Step 2 - Just run

.. code-block:: python

    experiment.run(port=8080)

现在，您已经成功启动了 NNI Experiment。 And you can type ``localhost:8080`` in your browser to observe your experiment in real time.

.. Note:: 实验将在前台运行，实验结束后自动退出。 If you want to run an experiment in an interactive way, use ``start()`` in Step 2. 

示例
^^^^^^^
以下是这种新的启动方法的示例。 你可以在 :githublink:`mnist-tfv2/launch.py <examples/trials/mnist-tfv2/launch.py>` 找到实验代码。

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

启动并管理一个新的 Experiment
------------------------------------------------------------------
我们将 ``NNI Client`` 中的 API 迁移到了这个新的启动方法。
通过 ``start()`` 而不是 ``run()`` 启动 Experiment，可以在交互模式下使用这些 API。

请参考 `示例用法 <./python_api_start.rst>`__ 和代码文件 :githublink:`python_api_start.ipynb <examples/trials/sklearn/classification/python_api_start.ipynb>`。

.. Note:: ``run()`` 轮询实验状态，并在实验完成时自动调用 ``stop()``。 ``start()`` 仅仅启动了一个新的 Experiment，所以需要通过调用 ``stop()`` 手动停止。

连接并管理已存在的 Experiment
----------------------------------------------------------------------------
如果您通过 ``nnictl`` 启动 Experiment，并且还想使用这些 API，那么可以使用 ``Experiment.connect()`` 连接到现有实验。

请参考 `示例用法 <./python_api_connect.rst>`__ 和代码文件 :githublink:`python_api_connect.ipynb <examples/trials/sklearn/classification/python_api_connect.ipynb>`。

.. Note:: 连接到现有 Experiment 时，可以使用 ``stop()`` 停止 Experiment。

API
---

..  autoclass:: nni.experiment.Experiment
    :members:
