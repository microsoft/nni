HPO Benchmarks
==============

..  toctree::
    :hidden:

    HPO Benchmark Example Statistics <hpo_benchmark_statistics>

We provide a benchmarking tool to compare the performances of tuners provided by NNI (and users' custom tuners) on different tasks. The implementation of this tool is based on the automlbenchmark repository (https://github.com/openml/automlbenchmark), which provides services of running different *frameworks* against different *benchmarks* consisting of multiple *tasks*. The tool is located in ``examples/trials/benchmarking/automlbenchmark``. This document provides a brief introduction to the tool and its usage. 

Terminology
^^^^^^^^^^^


* **task**\ : a task can be thought of as (dataset, evaluator). It gives out a dataset containing (train, valid, test), and based on the received predictions, the evaluator evaluates a given metric (e.g., mse for regression, f1 for classification). 
* **benchmark**\ : a benchmark is a set of tasks, along with other external constraints such as time and resource. 
* **framework**\ : given a task, a framework conceives answers to the proposed regression or classification problem and produces predictions. Note that the automlbenchmark framework does not pose any restrictions on the hypothesis space of a framework. In our implementation in this folder, each framework is a tuple (tuner, architecture), where architecture provides the hypothesis space (and search space for tuner), and tuner determines the strategy of hyperparameter optimization. 
* **tuner**\ : a tuner or advisor defined in the hpo folder, or a custom tuner provided by the user. 
* **architecture**\ : an architecture is a specific method for solving the tasks, along with a set of hyperparameters to optimize (i.e., the search space). In our implementation, the architecture calls tuner multiple times to obtain possible hyperparameter configurations, and produces the final prediction for a task. See ``./nni/extensions/NNI/architectures`` for examples.

Note: currently, the only architecture supported is random forest. The architecture implementation and search space definition can be found in ``./nni/extensions/NNI/architectures/run_random_forest.py``. The tasks in benchmarks "nnivalid" and "nnismall" are suitable to solve with random forests. 
  
Setup
^^^^^

Due to some incompatibilities between automlbenchmark and python 3.8, python 3.7 is recommended for running experiments contained in this folder. First, run the following shell script to clone the automlbenchmark repository. Note: it is recommended to perform the following steps in a separate virtual environment, as the setup code may install several packages. 

.. code-block:: bash

   ./setup.sh

Run predefined benchmarks on existing tuners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./runbenchmark_nni.sh [tuner-names]

This script runs the benchmark 'nnivalid', which consists of a regression task, a binary classification task, and a multi-class classification task. After the script finishes, you can find a summary of the results in the folder results_[time]/reports/. To run on other predefined benchmarks, change the ``benchmark`` variable in ``runbenchmark_nni.sh``. Some benchmarks are defined in ``/examples/trials/benchmarking/automlbenchmark/nni/benchmarks``\ , and others are defined in ``/examples/trials/benchmarking/automlbenchmark/automlbenchmark/resources/benchmarks/``. One example of larger benchmarks is "nnismall", which consists of 8 regression tasks, 8 binary classification tasks, and 8 multi-class classification tasks.

By default, the script runs the benchmark on all embedded tuners in NNI. If provided a list of tuners in [tuner-names], it only runs the tuners in the list. Currently, the following tuner names are supported: "TPE", "Random", "Anneal", "Evolution", "SMAC", "GPTuner", "MetisTuner", "DNGOTuner", "Hyperband", "BOHB". It is also possible to evaluate custom tuners. See the next sections for details. 

By default, the script runs the specified tuners against the specified benchmark one by one. To run all the experiments simultaneously in the background, set the "serialize" flag to false in ``runbenchmark_nni.sh``. 

Note: the SMAC tuner, DNGO tuner, and the BOHB advisor has to be manually installed before any experiments can be run on it. Please refer to `this page <https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html?highlight=nni>`_ for more details on installing SMAC and BOHB.

Run customized benchmarks on existing tuners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run customized benchmarks, add a benchmark_name.yaml file in the folder ``./nni/benchmarks``\ , and change the ``benchmark`` variable in ``runbenchmark_nni.sh``. See ``./automlbenchmark/resources/benchmarks/`` for some examples of defining a custom benchmark.

Run benchmarks on custom tuners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom tuners, first make sure that the tuner inherits from ``nni.tuner.Tuner`` and correctly implements the required APIs. For more information on implementing a custom tuner, please refer to `here <https://nni.readthedocs.io/en/stable/Tuner/CustomizeTuner.html>`_. Next, perform the following steps:


#. Install the custom tuner with command ``nnictl algo register``. Check `this document <https://nni.readthedocs.io/en/stable/Tutorial/Nnictl.html>`_ for details. 
#. In ``./nni/frameworks.yaml``\ , add a new framework extending the base framework NNI. Make sure that the parameter ``tuner_type`` corresponds to the "builtinName" of tuner installed in step 1.
#. Run the following command

.. code-block:: bash

      ./runbenchmark_nni.sh new-tuner-builtinName
