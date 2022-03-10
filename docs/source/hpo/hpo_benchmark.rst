HPO Benchmarks
==============

..  toctree::
    :hidden:

    HPO Benchmark Example Statistics <hpo_benchmark_stats>

We provide a benchmarking tool to compare the performances of tuners provided by NNI (and users' custom tuners) on different
types of tasks. This tool uses the `automlbenchmark repository <https://github.com/openml/automlbenchmark)>`_  to run different *benchmarks* on the NNI *tuners*.
The tool is located in ``examples/trials/benchmarking/automlbenchmark``. This document provides a brief introduction to the tool, its usage, and currently available benchmarks.

Overview and Terminologies
^^^^^^^^^^^^^^^^^^^^^^^^^^

Ideally, an **HPO Benchmark** provides a tuner with a search space, calls the tuner repeatedly, and evaluates how the tuner probes
the search space and approaches to good solutions. In addition, inside the benchmark, an evaluator should be associated to
each search space for evaluating the score of points in this search space to give feedbacks to the tuner. For instance,
the search space could be the space of hyperparameters for a neural network. Then the evaluator should contain train data,
test data, and a criterion. To evaluate a point in the search space, the evaluator will train the network on the train data
and report the score of the model on the test data as the score for the point.

However, a **benchmark** provided by the automlbenchmark repository only provides part of the functionality of the evaluator.
More concretely, it assumes that it is evaluating a **framework**. Different from a tuner, given train data, a **framework**
can directly solve a **task** and predict on the test set. The **benchmark** from the automlbenchmark repository directly provides
train and test datasets to a **framework**, evaluates the prediction on the test set, and reports this score as the final score.
Therefore, to implement **HPO Benchmark** using automlbenchmark, we pair up a tuner with a search space to form a **framework**,
and handle the repeated trial-evaluate-feedback loop in the **framework** abstraction. In other words, each **HPO Benchmark**
contains two main components: a **benchmark** from the automlbenchmark library, and an **architecture** which defines the search
space and the evaluator. To further clarify, we provide the definition for the terminologies used in this document.

* **tuner**\ : a `tuner or advisor provided by NNI <https://nni.readthedocs.io/en/stable/builtin_tuner.html>`_, or a custom tuner provided by the user.
* **task**\ : an abstraction used by automlbenchmark. A task can be thought of as a tuple (dataset, metric). It provides train and test datasets to the frameworks. Then, based on the returns predictions on the test set, the task evaluates the metric (e.g., mse for regression, f1 for classification) and reports the score.
* **benchmark**\ : an abstraction used by automlbenchmark. A benchmark is a set of tasks, along with other external constraints such as time limits.
* **framework**\ : an abstraction used by automlbenchmark. Given a task, a framework solves the proposed regression or classification problem using train data and produces predictions on the test set. In our implementation, each framework is an architecture, which defines a search space. To evaluate a task given by the benchmark on a specific tuner, we let the tuner continuously tune the hyperparameters (by giving it cross-validation score on the train data as feedback) until the time or trial limit is reached. Then, the architecture is retrained on the entire train set using the best set of hyperparameters.
* **architecture**\ : an architecture is a specific method for solving the tasks, along with a set of hyperparameters to optimize (i.e., the search space). See ``./nni/extensions/NNI/architectures`` for examples.

Supported HPO Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^

From the previous discussion, we can see that to define an **HPO Benchmark**, we need to specify a **benchmark** and an **architecture**.

Currently, the only architectures we support are random forest and MLP. We use the
`scikit-learn implementation <https://scikit-learn.org/stable/modules/classes.html#>`_. Typically, there are a number of
hyperparameters that may directly affect the performances of random forest and MLP models. We design the search
spaces to be the following.

Search Space for Random Forest:

.. code-block:: json

   {
       "n_estimators": {"_type":"randint", "_value": [4, 2048]},
       "max_depth": {"_type":"choice", "_value": [4, 8, 16, 32, 64, 128, 256, 0]},
       "min_samples_leaf": {"_type":"randint", "_value": [1, 8]},
       "min_samples_split": {"_type":"randint", "_value": [2, 16]},
       "max_leaf_nodes": {"_type":"randint", "_value": [0, 4096]}
    }

Search Space for MLP:

.. code-block:: json

    {
       "hidden_layer_sizes": {"_type":"choice", "_value": [[16], [64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256], [16, 16, 16], [64, 64, 64], [128, 128, 128], [256, 256, 256], [256, 128, 64, 16], [128, 64, 16], [64, 16], [16, 64, 128, 256], [16, 64, 128], [16, 64]]},
       "learning_rate_init": {"_type":"choice", "_value": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]},
       "alpha": {"_type":"choice", "_value": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
       "momentum": {"_type":"uniform","_value":[0, 1]},
       "beta_1": {"_type":"uniform","_value":[0, 1]},
       "tol": {"_type":"choice", "_value": [0.001, 0.0005, 0.0001, 0.00005, 0.00001]},
       "max_iter": {"_type":"randint", "_value": [2, 256]}
    }

In addition, we write the search space in different ways (e.g., using "choice" or "randint" or "loguniform").
The architecture implementation and search space definition can be found in ``./nni/extensions/NNI/architectures/``.
You may replace the search space definition in this file to experiment different search spaces.

For the automlbenchmarks, in addition to the built-in benchmarks provided by automl
(defined in ``/examples/trials/benchmarking/automlbenchmark/automlbenchmark/resources/benchmarks/``), we design several
additional benchmarks, defined in ``/examples/trials/benchmarking/automlbenchmark/nni/benchmarks``.
One example of larger benchmarks is "nnismall", which consists of 8 regression tasks, 8 binary classification tasks, and
8 multi-class classification tasks. We also provide three separate 8-task benchmarks "nnismall-regression", "nnismall-binary", and "nnismall-multiclass"
corresponding to the three types of tasks in nnismall. These tasks are suitable to solve with random forest and MLP.

The following table summarizes the benchmarks we provide. For ``nnismall``, please check ``/examples/trials/benchmarking/automlbenchmark/automlbenchmark/resources/benchmarks/``
for a more detailed description for each task. Also, since all tasks are from the OpenML platform, you can find the descriptions
of all datasets at `this webpage <https://www.openml.org/search?type=data>`_.

.. list-table::
   :header-rows: 1
   :widths: 1 2 2 2

   * - Benchmark name
     - Description
     - Task List
     - Location
   * - nnivalid
     - A three-task benchmark to validate benchmark installation.
     - ``kc2, iris, cholesterol``
     - ``/examples/trials/benchmarking/automlbenchmark/nni/benchmarks/``
   * - nnismall-regression
     - An eight-task benchmark consisting of **regression** tasks only.
     - ``cholesterol, liver-disorders, kin8nm, cpu_small, titanic_2, boston, stock, space_ga``
     - ``/examples/trials/benchmarking/automlbenchmark/nni/benchmarks/``
   * - nnismall-binary
     - An eight-task benchmark consisting of **binary classification** tasks only.
     - ``Australian, blood-transfusion, christine, credit-g, kc1, kr-vs-kp, phoneme, sylvine``
     - ``/examples/trials/benchmarking/automlbenchmark/nni/benchmarks/``
   * - nnismall-multiclass
     - An eight-task benchmark consisting of **multi-class classification** tasks only.
     - ``car, cnae-9, dilbert, fabert, jasmine, mfeat-factors, segment, vehicle``
     - ``/examples/trials/benchmarking/automlbenchmark/nni/benchmarks/``
   * - nnismall
     - A 24-task benchmark that is the superset of nnismall-regression, nnismall-binary, and nnismall-multiclass.
     - ``cholesterol, liver-disorders, kin8nm, cpu_small, titanic_2, boston, stock, space_ga, Australian, blood-transfusion, christine, credit-g, kc1, kr-vs-kp, phoneme, sylvine, car, cnae-9, dilbert, fabert, jasmine, mfeat-factors, segment, vehicle``
     - ``/examples/trials/benchmarking/automlbenchmark/nni/benchmarks/``

Setup
^^^^^

Due to some incompatibilities between automlbenchmark and python 3.8, python 3.7 is recommended for running experiments contained in this folder. First, run the following shell script to clone the automlbenchmark repository. Note: it is recommended to perform the following steps in a separate virtual environment, as the setup code may install several packages. 

.. code-block:: bash

   ./setup.sh

Run predefined benchmarks on existing tuners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./runbenchmark_nni.sh [tuner-names]

This script runs the benchmark 'nnivalid', which consists of a regression task, a binary classification task, and a
multi-class classification task. After the script finishes, you can find a summary of the results in the folder results_[time]/reports/.
To run on other predefined benchmarks, change the ``benchmark`` variable in ``runbenchmark_nni.sh``. To change to another
search space (by using another architecture), chang the `arch_type` parameter in ``./nni/frameworks.yaml``. Note that currently,
we only support ``random_forest`` or ``mlp`` as the `arch_type`. To experiment on other search spaces with the same
architecture, please change the search space defined in ``./nni/extensions/NNI/architectures/run_[architecture].py``.

The ``./nni/frameworks.yaml`` is the actual configuration file for the HPO Benchmark. The ``limit_type`` parameter specifies
the limits for running the benchmark on one tuner. If ``limit_type`` is set to `ntrials`, then the tuner is called for
`trial_limit` times and then stopped. If ``limit_type`` is set to `time`, then the tuner is continuously called until
timeout for the benchmark is reached. The timeout for the benchmarks can be changed in the each benchmark file located
in ``./nni/benchmarks``.

By default, the script runs the benchmark on all embedded tuners in NNI. If provided a list of tuners in [tuner-names],
it only runs the tuners in the list. Currently, the following tuner names are supported: "TPE", "Random", "Anneal",
"Evolution", "SMAC", "GPTuner", "MetisTuner", "DNGOTuner", "Hyperband", "BOHB". It is also possible to run the benchmark
on custom tuners. See the next sections for details.

By default, the script runs the specified tuners against the specified benchmark one by one. To run the experiment for
all tuners simultaneously in the background, set the "serialize" flag to false in ``runbenchmark_nni.sh``.

Note: the SMAC tuner, DNGO tuner, and the BOHB advisor has to be manually installed before running benchmarks on them.
Please refer to `this page <https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html?highlight=nni>`_ for more details
on installation.

Run customized benchmarks on existing tuners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can design your own benchmarks and evaluate the performance of NNI tuners on them. To run customized benchmarks,
add a benchmark_name.yaml file in the folder ``./nni/benchmarks``, and change the ``benchmark`` variable in ``runbenchmark_nni.sh``.
See ``./automlbenchmark/resources/benchmarks/`` for some examples of defining a custom benchmark.

Run benchmarks on custom tuners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may also use the benchmark to compare a custom tuner written by yourself with the NNI built-in tuners. To use custom
tuners, first make sure that the tuner inherits from ``nni.tuner.Tuner`` and correctly implements the required APIs. For
more information on implementing a custom tuner, please refer to `here <https://nni.readthedocs.io/en/stable/Tuner/CustomizeTuner.html>`_.
Next, perform the following steps:

#. Install the custom tuner via the command ``nnictl algo register``. Check `this document <https://nni.readthedocs.io/en/stable/Tutorial/Nnictl.html>`_ for details. 
#. In ``./nni/frameworks.yaml``\ , add a new framework extending the base framework NNI. Make sure that the parameter ``tuner_type`` corresponds to the "builtinName" of tuner installed in step 1.
#. Run the following command

.. code-block:: bash

      ./runbenchmark_nni.sh new-tuner-builtinName

The benchmark will automatically find and match the tuner newly added to your NNI installation.
