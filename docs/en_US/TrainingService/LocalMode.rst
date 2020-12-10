**Tutorial: Create and Run an Experiment on local with NNI API**
====================================================================

In this tutorial, we will use the example in [~/examples/trials/mnist-tfv1] to explain how to create and run an experiment on local with NNI API.

..

   Before starts


You have an implementation for MNIST classifer using convolutional layers, the Python code is in ``mnist_before.py``.

..

   Step 1 - Update model codes


To enable NNI API, make the following changes:

* Declare NNI API: include ``import nni`` in your trial code to use NNI APIs.
* Get predefined parameters

Use the following code snippet:

.. code-block:: python

   RECEIVED_PARAMS = nni.get_next_parameter()

to get hyper-parameters' values assigned by tuner. ``RECEIVED_PARAMS`` is an object, for example:

.. code-block:: json

   {"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}

* Report NNI results: Use the API: ``nni.report_intermediate_result(accuracy)`` to send ``accuracy`` to assessor.
  Use the API: ``nni.report_final_result(accuracy)`` to send `accuracy` to tuner.

We had made the changes and saved it to ``mnist.py``.

**NOTE**\ :

.. code-block:: bash

   accuracy - The `accuracy` could be any python object, but  if you use NNI built-in tuner/assessor, `accuracy` should be a numerical variable (e.g. float, int).
   assessor - The assessor will decide which trial should early stop based on the history performance of trial (intermediate result of one trial).
   tuner    - The tuner will generate next parameters/architecture based on the explore history (final result of all trials).

..

   Step 2 - Define SearchSpace


The hyper-parameters used in ``Step 1.2 - Get predefined parameters`` is defined in a ``search_space.json`` file like below:

.. code-block:: bash

   {
       "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
       "conv_size":{"_type":"choice","_value":[2,3,5,7]},
       "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
       "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
   }

Refer to `define search space <../Tutorial/SearchSpaceSpec.rst>`__ to learn more about search space.

..

   Step 3 - Define Experiment

   ..

      3.1 enable NNI API mode


To enable NNI API mode, you need to set useAnnotation to *false* and provide the path of SearchSpace file (you just defined in step 1):

.. code-block:: bash

   useAnnotation: false
   searchSpacePath: /path/to/your/search_space.json

To run an experiment in NNI, you only needed:


* Provide a runnable trial
* Provide or choose a tuner
* Provide a YAML experiment configure file
* (optional) Provide or choose an assessor

**Prepare trial**\ :

..

   A set of examples can be found in ~/nni/examples after your installation, run ``ls ~/nni/examples/trials`` to see all the trial examples.


Let's use a simple trial example, e.g. mnist, provided by NNI. After you installed NNI, NNI examples have been put in ~/nni/examples, run ``ls ~/nni/examples/trials`` to see all the trial examples. You can simply execute the following command to run the NNI mnist example:

.. code-block:: bash

     python ~/nni/examples/trials/mnist-annotation/mnist.py


This command will be filled in the YAML configure file below. Please refer to `here <../TrialExample/Trials.rst>`__ for how to write your own trial.

**Prepare tuner**\ : NNI supports several popular automl algorithms, including Random Search, Tree of Parzen Estimators (TPE), Evolution algorithm etc. Users can write their own tuner (refer to `here <../Tuner/CustomizeTuner.rst>`__\ ), but for simplicity, here we choose a tuner provided by NNI as below:

.. code-block:: bash

     tuner:
       builtinTunerName: TPE
       classArgs:
         optimize_mode: maximize


*builtinTunerName* is used to specify a tuner in NNI, *classArgs* are the arguments pass to the tuner (the spec of builtin tuners can be found `here <../Tuner/BuiltinTuner.rst>`__\ ), *optimization_mode* is to indicate whether you want to maximize or minimize your trial's result.

**Prepare configure file**\ : Since you have already known which trial code you are going to run and which tuner you are going to use, it is time to prepare the YAML configure file. NNI provides a demo configure file for each trial example, ``cat ~/nni/examples/trials/mnist-annotation/config.yml`` to see it. Its content is basically shown below:

.. code-block:: yaml

   authorName: your_name
   experimentName: auto_mnist

   # how many trials could be concurrently running
   trialConcurrency: 1

   # maximum experiment running duration
   maxExecDuration: 3h

   # empty means never stop
   maxTrialNum: 100

   # choice: local, remote
   trainingServicePlatform: local

   # search space file
   searchSpacePath: search_space.json

   # choice: true, false
   useAnnotation: true
   tuner:
     builtinTunerName: TPE
     classArgs:
       optimize_mode: maximize
   trial:
     command: python mnist.py
     codeDir: ~/nni/examples/trials/mnist-annotation
     gpuNum: 0

Here *useAnnotation* is true because this trial example uses our python annotation (refer to `here <../Tutorial/AnnotationSpec.rst>`__ for details). For trial, we should provide *trialCommand* which is the command to run the trial, provide *trialCodeDir* where the trial code is. The command will be executed in this directory. We should also provide how many GPUs a trial requires.

With all these steps done, we can run the experiment with the following command:

.. code-block:: bash

     nnictl create --config ~/nni/examples/trials/mnist-annotation/config.yml


You can refer to `here <../Tutorial/Nnictl.rst>`__ for more usage guide of *nnictl* command line tool.

View experiment results
-----------------------

The experiment has been running now. Other than *nnictl*\ , NNI also provides WebUI for you to view experiment progress, to control your experiment, and some other appealing features.

Using multiple local GPUs to speed up search
--------------------------------------------

The following steps assume that you have 4 NVIDIA GPUs installed at local and `tensorflow with GPU support <https://www.tensorflow.org/install/gpu>`__. The demo enables 4 concurrent trail jobs and each trail job uses 1 GPU.

**Prepare configure file**\ : NNI provides a demo configuration file for the setting above, ``cat ~/nni/examples/trials/mnist-annotation/config_gpu.yml`` to see it. The trailConcurrency and gpuNum are different from the basic configure file:

.. code-block:: bash

   ...

   # how many trials could be concurrently running
   trialConcurrency: 4

   ...

   trial:
     command: python mnist.py
     codeDir: ~/nni/examples/trials/mnist-annotation
     gpuNum: 1

We can run the experiment with the following command:

.. code-block:: bash

     nnictl create --config ~/nni/examples/trials/mnist-annotation/config_gpu.yml


You can use *nnictl* command line tool or WebUI to trace the training progress. *nvidia_smi* command line tool can also help you to monitor the GPU usage during training.
