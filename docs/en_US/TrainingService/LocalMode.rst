**Tutorial: Create and Run an Experiment on local with NNI API**
================================================================

In this tutorial, we will use the example in [nni/examples/trials/mnist-pytorch] to explain how to create and run an experiment on local with NNI API.

..

   Before starts


You have an implementation for MNIST classifer using convolutional layers, the Python code is similar to ``mnist.py``.

..

   Step 1 - Update model codes


To enable NNI API, make the following changes:

1.1 Declare NNI API: include ``import nni`` in your trial code to use NNI APIs.

1.2 Get predefined parameters

Use the following code snippet:

.. code-block:: python

   tuner_params = nni.get_next_parameter()

to get hyper-parameters' values assigned by tuner. ``tuner_params`` is an object, for example:

.. code-block:: json

   {"batch_size": 32, "hidden_size": 128, "lr": 0.01, "momentum": 0.2029}

..

1.3 Report NNI results: Use the API: ``nni.report_intermediate_result(accuracy)`` to send ``accuracy`` to assessor. Use the API: ``nni.report_final_result(accuracy)`` to send `accuracy` to tuner.

**NOTE**\ :

.. code-block:: bash

   accuracy - The `accuracy` could be any python object, but  if you use NNI built-in tuner/assessor, `accuracy` should be a numerical variable (e.g. float, int).
   tuner    - The tuner will generate next parameters/architecture based on the explore history (final result of all trials).
   assessor - The assessor will decide which trial should early stop based on the history performance of trial (intermediate result of one trial).

..

   Step 2 - Define SearchSpace


The hyper-parameters used in ``Step 1.2 - Get predefined parameters`` is defined in a ``search_space.json`` file like below:

.. code-block:: bash

    {
        "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
        "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
        "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
        "momentum":{"_type":"uniform","_value":[0, 1]}
    }

Refer to `define search space <../Tutorial/SearchSpaceSpec.rst>`__ to learn more about search space.

..

   Step 3 - Define Experiment

   ..

To run an experiment in NNI, you only needed:


* Provide a runnable trial
* Provide or choose a tuner
* Provide a YAML experiment configure file
* (optional) Provide or choose an assessor

**Prepare trial**\ :

..

   You can download nni source code and a set of examples can be found in ``nni/examples``, run ``ls nni/examples/trials`` to see all the trial examples.


Let's use a simple trial example, e.g. mnist, provided by NNI. After you cloned NNI source, NNI examples have been put in ~/nni/examples, run ``ls ~/nni/examples/trials`` to see all the trial examples. You can simply execute the following command to run the NNI mnist example:

.. code-block:: bash

     python ~/nni/examples/trials/mnist-pytorch/mnist.py


This command will be filled in the YAML configure file below. Please refer to `here <../TrialExample/Trials.rst>`__ for how to write your own trial.

**Prepare tuner**\ : NNI supports several popular automl algorithms, including Random Search, Tree of Parzen Estimators (TPE), Evolution algorithm etc. Users can write their own tuner (refer to `here <../Tuner/CustomizeTuner.rst>`__\ ), but for simplicity, here we choose a tuner provided by NNI as below:

.. code-block:: bash

     tuner:
       name: TPE
       classArgs:
         optimize_mode: maximize


*name* is used to specify a tuner in NNI, *classArgs* are the arguments pass to the tuner (the spec of builtin tuners can be found `here <../Tuner/BuiltinTuner.rst>`__\ ), *optimization_mode* is to indicate whether you want to maximize or minimize your trial's result.

**Prepare configure file**\ : Since you have already known which trial code you are going to run and which tuner you are going to use, it is time to prepare the YAML configure file. NNI provides a demo configure file for each trial example, ``cat ~/nni/examples/trials/mnist-pytorch/config.yml`` to see it. Its content is basically shown below:

.. code-block:: yaml

   experimentName: local training service example

   searchSpaceFile ~/nni/examples/trials/mnist-pytorch/search_space.json
   trailCommand: python3 mnist.py
   trialCodeDirectory: ~/nni/examples/trials/mnist-pytorch

   trialGpuNumber: 0
   trialConcurrency: 1
   maxExperimentDuration: 3h
   maxTrialNumber: 10

   trainingService:
     platform: local

   tuner:
     name: TPE
     classArgs:
       optimize_mode: maximize


With all these steps done, we can run the experiment with the following command:

.. code-block:: bash

     nnictl create --config ~/nni/examples/trials/mnist-pytorch/config.yml


You can refer to `here <../Tutorial/Nnictl.rst>`__ for more usage guide of *nnictl* command line tool.

View experiment results
-----------------------

The experiment has been running now. Other than *nnictl*\ , NNI also provides WebUI for you to view experiment progress, to control your experiment, and some other appealing features.

Using multiple local GPUs to speed up search
--------------------------------------------

The following steps assume that you have 4 NVIDIA GPUs installed at local and PyTorch with CUDA support. The demo enables 4 concurrent trail jobs and each trail job uses 1 GPU.

**Prepare configure file**\ : NNI provides a demo configuration file for the setting above, ``cat ~/nni/examples/trials/mnist-pytorch/config_detailed.yml`` to see it. The trailConcurrency and trialGpuNumber are different from the basic configure file:

.. code-block:: bash

   ...

   trialGpuNumber: 1
   trialConcurrency: 4

   ...

   trainingService:
     platform: local
     useActiveGpu: false  # set to "true" if you are using graphical OS like Windows 10 and Ubuntu desktop


We can run the experiment with the following command:

.. code-block:: bash

     nnictl create --config ~/nni/examples/trials/mnist-pytorch/config_detailed.yml


You can use *nnictl* command line tool or WebUI to trace the training progress. *nvidia_smi* command line tool can also help you to monitor the GPU usage during training.
