.. role:: raw-html(raw)
   :format: html


Write a Trial Run on NNI
========================

A **Trial** in NNI is an individual attempt at applying a configuration (e.g., a set of hyper-parameters) to a model.

To define an NNI trial, you need to first define the set of parameters (i.e., search space) and then update the model. NNI provides two approaches for you to define a trial: `NNI API <#nni-api>`__ and `NNI Python annotation <#nni-annotation>`__. You could also refer to `here <#more-examples>`__ for more trial examples.

:raw-html:`<a name="nni-api"></a>`

NNI API
-------

Step 1 - Prepare a SearchSpace parameters file.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example is shown below:

.. code-block:: json

   {
       "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
       "conv_size":{"_type":"choice","_value":[2,3,5,7]},
       "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
       "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
   }

Refer to `SearchSpaceSpec.md <../Tutorial/SearchSpaceSpec.rst>`__ to learn more about search spaces. Tuner will generate configurations from this search space, that is, choosing a value for each hyperparameter from the range.

Step 2 - Update model code
^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  Import NNI

    Include ``import nni`` in your trial code to use NNI APIs.

* 
  Get configuration from Tuner

.. code-block:: python

   RECEIVED_PARAMS = nni.get_next_parameter()

``RECEIVED_PARAMS`` is an object, for example:

``{"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}``.


* Report metric data periodically (optional)

.. code-block:: python

   nni.report_intermediate_result(metrics)

``metrics`` can be any python object. If users use the NNI built-in tuner/assessor, ``metrics`` can only have two formats: 1) a number e.g., float, int, or 2) a dict object that has a key named ``default`` whose value is a number. These ``metrics`` are reported to `assessor <../Assessor/BuiltinAssessor.rst>`__. Often, ``metrics`` includes the periodically evaluated loss or accuracy.


* Report performance of the configuration

.. code-block:: python

   nni.report_final_result(metrics)

``metrics`` can also be any python object. If users use the NNI built-in tuner/assessor, ``metrics`` follows the same format rule as that in ``report_intermediate_result``\ , the number indicates the model's performance, for example, the model's accuracy, loss etc. These ``metrics`` are reported to `tuner <../Tuner/BuiltinTuner.rst>`__.

Step 3 - Enable NNI API
^^^^^^^^^^^^^^^^^^^^^^^

To enable NNI API mode, you need to set useAnnotation to *false* and provide the path of the SearchSpace file was defined in step 1:

.. code-block:: yaml

   useAnnotation: false
   searchSpacePath: /path/to/your/search_space.json

You can refer to `here <../Tutorial/ExperimentConfig.rst>`__ for more information about how to set up experiment configurations.

Please refer to `here </sdk_reference.html>`__ for more APIs (e.g., ``nni.get_sequence_id()``\ ) provided by NNI.

:raw-html:`<a name="nni-annotation"></a>`

NNI Python Annotation
---------------------

An alternative to writing a trial is to use NNI's syntax for python. NNI annotations are simple, similar to comments. You don't have to make structural changes to your existing code. With a few lines of NNI annotation, you will be able to:


* annotate the variables you want to tune
* specify the range  in which you want to tune the variables
* annotate which variable you want to report as an intermediate result to ``assessor``
* annotate which variable you want to report as the final result (e.g. model accuracy) to ``tuner``.

Again, take MNIST as an example, it only requires 2 steps to write a trial with NNI Annotation.

Step 1 - Update codes with annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a TensorFlow code snippet for NNI Annotation where the highlighted four lines are annotations that:


#. tune batch_size and dropout_rate
#. report test_acc every 100 steps
#. lastly report test_acc as the final result.

It's worth noting that, as these newly added codes are merely annotations, you can still run your code as usual in environments without NNI installed.

.. code-block:: diff

   with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
   +   """@nni.variable(nni.choice(50, 250, 500), name=batch_size)"""
       batch_size = 128
       for i in range(10000):
           batch = mnist.train.next_batch(batch_size)
   +       """@nni.variable(nni.choice(0.1, 0.5), name=dropout_rate)"""
           dropout_rate = 0.5
           mnist_network.train_step.run(feed_dict={mnist_network.images: batch[0],
                                                   mnist_network.labels: batch[1],
                                                   mnist_network.keep_prob: dropout_rate})
           if i % 100 == 0:
               test_acc = mnist_network.accuracy.eval(
                   feed_dict={mnist_network.images: mnist.test.images,
                               mnist_network.labels: mnist.test.labels,
                               mnist_network.keep_prob: 1.0})
   +           """@nni.report_intermediate_result(test_acc)"""

       test_acc = mnist_network.accuracy.eval(
           feed_dict={mnist_network.images: mnist.test.images,
                       mnist_network.labels: mnist.test.labels,
                       mnist_network.keep_prob: 1.0})
   +   """@nni.report_final_result(test_acc)"""

**NOTE**\ :


* ``@nni.variable`` will affect its following line which should be an assignment statement whose left-hand side must be the same as the keyword ``name`` in the ``@nni.variable`` statement.
* ``@nni.report_intermediate_result``\ /\ ``@nni.report_final_result`` will send the data to assessor/tuner at that line.

For more information about annotation syntax and its usage, please refer to `Annotation <../Tutorial/AnnotationSpec.rst>`__.

Step 2 - Enable NNI Annotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the YAML configure file, you need to set *useAnnotation* to true to enable NNI annotation:

.. code-block:: bash

   useAnnotation: true

Standalone mode for debugging
-----------------------------

NNI supports a standalone mode for trial code to run without starting an NNI experiment. This is for finding out bugs in trial code more conveniently. NNI annotation natively supports standalone mode, as the added NNI related lines are comments. For NNI trial APIs, the APIs have changed behaviors in standalone mode, some APIs return dummy values, and some APIs do not really report values. Please refer to the following table for the full list of these APIs.

.. code-block:: python

   # NOTE: please assign default values to the hyperparameters in your trial code
   nni.get_next_parameter # return {}
   nni.report_final_result # have log printed on stdout, but does not report
   nni.report_intermediate_result # have log printed on stdout, but does not report
   nni.get_experiment_id # return "STANDALONE"
   nni.get_trial_id # return "STANDALONE"
   nni.get_sequence_id # return 0

You can try standalone mode with the :githublink:`mnist example <examples/trials/mnist-tfv1>`. Simply run ``python3 mnist.py`` under the code directory. The trial code should successfully run with the default hyperparameter values.

For more information on debugging, please refer to `How to Debug <../Tutorial/HowToDebug.rst>`__

Where are my trials?
--------------------

Local Mode
^^^^^^^^^^

In NNI, every trial has a dedicated directory for them to output their own data. In each trial, an environment variable called ``NNI_OUTPUT_DIR`` is exported. Under this directory, you can find each trial's code, data, and other logs. In addition, each trial's log (including stdout) will be re-directed to a file named ``trial.log`` under that directory.

If NNI Annotation is used, the trial's converted code is in another temporary directory. You can check that in a file named ``run.sh`` under the directory indicated by ``NNI_OUTPUT_DIR``. The second line (i.e., the ``cd`` command) of this file will change directory to the actual directory where code is located. Below is an example of ``run.sh``\ :

.. code-block:: bash

   #!/bin/bash
   cd /tmp/user_name/nni/annotation/tmpzj0h72x6 #This is the actual directory
   export NNI_PLATFORM=local
   export NNI_SYS_DIR=/home/user_name/nni-experiments/$experiment_id$/trials/$trial_id$
   export NNI_TRIAL_JOB_ID=nrbb2
   export NNI_OUTPUT_DIR=/home/user_name/nni-experiments/$eperiment_id$/trials/$trial_id$
   export NNI_TRIAL_SEQ_ID=1
   export MULTI_PHASE=false
   export CUDA_VISIBLE_DEVICES=
   eval python3 mnist.py 2>/home/user_name/nni-experiments/$experiment_id$/trials/$trial_id$/stderr
   echo $? `date +%s%3N` >/home/user_name/nni-experiments/$experiment_id$/trials/$trial_id$/.nni/state

Other Modes
^^^^^^^^^^^

When running trials on other platforms like remote machine or PAI, the environment variable ``NNI_OUTPUT_DIR`` only refers to the output directory of the trial, while the trial code and ``run.sh`` might not be there. However, the ``trial.log`` will be transmitted back to the local machine in the trial's directory, which defaults to ``~/nni-experiments/$experiment_id$/trials/$trial_id$/``

For more information, please refer to `HowToDebug <../Tutorial/HowToDebug.rst>`__.

:raw-html:`<a name="more-examples"></a>`

More Trial Examples
-------------------


* `MNIST examples <MnistExamples.rst>`__
* `Finding out best optimizer for Cifar10 classification <Cifar10Examples.rst>`__
* `How to tune Scikit-learn on NNI <SklearnExamples.rst>`__
* `Automatic Model Architecture Search for Reading Comprehension. <SquadEvolutionExamples.rst>`__
* `Tuning GBDT on NNI <GbdtExample.rst>`__
* `Tuning RocksDB on NNI <RocksdbExamples.rst>`__
