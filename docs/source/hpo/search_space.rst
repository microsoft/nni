Search Space
============

Overview
--------

In NNI, tuner will sample hyperparameters according to the search space.

To define a search space, users should define the name of the variable, the type of sampling strategy and its parameters.

* An example of a search space definition in JSON format is as follow:

.. code-block:: json 

   {
       "dropout_rate": {"_type": "uniform", "_value": [0.1, 0.5]},
       "conv_size": {"_type": "choice", "_value": [2, 3, 5, 7]},
       "hidden_size": {"_type": "choice", "_value": [124, 512, 1024]},
       "batch_size": {"_type": "choice", "_value": [50, 250, 500]},
       "learning_rate": {"_type": "uniform", "_value": [0.0001, 0.1]}
   }

Take the first line as an example.
``dropout_rate`` is defined as a variable whose prior distribution is a uniform distribution with a range from ``0.1`` to ``0.5``.

.. attention::

    The available sampling strategies within a search space depend on the tuner you want to use.
    We list the supported types for each built-in tuner :ref:`below <hpo-space-support>`.

    For a customized tuner, you don't have to follow our convention and you will have the flexibility to define any type you want.

Types
-----

All types of sampling strategies and their parameter are listed here:

choice
^^^^^^

.. code-block:: python 

    {"_type": "choice", "_value": options}

* The variable's value is one of the options. Here ``options`` should be a list of **numbers** or a list of **strings**. Using arbitrary objects as members of this list (like sublists, a mixture of numbers and strings, or null values) should work in most cases, but may trigger undefined behaviors.
* ``options`` can also be a nested sub-search-space, this sub-search-space takes effect only when the corresponding element is chosen. The variables in this sub-search-space can be seen as conditional variables. Here is an simple :githublink:`example of nested search space definition <examples/trials/mnist-nested-search-space/search_space.json>`. If an element in the options list is a dict, it is a sub-search-space, and for our built-in tuners you have to add a ``_name`` key in this dict, which helps you to identify which element is chosen. Accordingly, here is a :githublink:`sample <examples/trials/mnist-nested-search-space/sample.json>` which users can get from nni with nested search space definition. See the table below for the tuners which support nested search spaces.

randint
^^^^^^^

.. code-block:: python 

    {"_type": "randint", "_value": [lower, upper]}

* Choosing a random integer between ``lower`` (inclusive) and ``upper`` (exclusive).
* Note: Different tuners may interpret ``randint`` differently. Some (e.g., TPE, GridSearch) treat integers from lower
  to upper as unordered ones, while others respect the ordering (e.g., SMAC). If you want all the tuners to respect
  the ordering, please use ``quniform`` with ``q=1``.

uniform
^^^^^^^

.. code-block:: python 

    {"_type": "uniform", "_value": [low, high]}

* The variable value is uniformly sampled between low and high.
* When optimizing, this variable is constrained to a two-sided interval.

quniform
^^^^^^^^

.. code-block:: python 

    {"_type": "quniform", "_value": [low, high, q]}

* The variable value is determined using ``clip(round(uniform(low, high) / q) * q, low, high)``\ , where the clip operation is used to constrain the generated value within the bounds. For example, for ``_value`` specified as [0, 10, 2.5], possible values are [0, 2.5, 5.0, 7.5, 10.0]; For ``_value`` specified as [2, 10, 5], possible values are [2, 5, 10].
* Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below. If you want to uniformly choose an integer from a range [low, high], you can write ``_value`` like this: ``[low, high, 1]``.

loguniform
^^^^^^^^^^

.. code-block:: python 

    {"_type": "loguniform", "_value": [low, high]}

* The variable value is drawn from a range [low, high] according to a loguniform distribution like exp(uniform(log(low), log(high))), so that the logarithm of the return value is uniformly distributed.
* When optimizing, this variable is constrained to be positive.

qloguniform
^^^^^^^^^^^

.. code-block:: python 

    {"_type": "qloguniform", "_value": [low, high, q]}

* The variable value is determined using ``clip(round(loguniform(low, high) / q) * q, low, high)``\ , where the clip operation is used to constrain the generated value within the bounds.
* Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.

normal
^^^^^^

.. code-block:: python 

    {"_type": "normal", "_value": [mu, sigma]}

* The variable value is a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.

qnormal
^^^^^^^

.. code-block:: python 

    {"_type": "qnormal", "_value": [mu, sigma, q]}

* The variable value is determined using ``round(normal(mu, sigma) / q) * q``
* Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.

lognormal
^^^^^^^^^

.. code-block:: python 

    {"_type": "lognormal", "_value": [mu, sigma]}

* The variable value is drawn according to ``exp(normal(mu, sigma))`` so that the logarithm of the return value is normally distributed. When optimizing, this variable is constrained to be positive.

qlognormal
^^^^^^^^^^

.. code-block:: python 

    {"_type": "qlognormal", "_value": [mu, sigma, q]}

* The variable value is determined using ``round(exp(normal(mu, sigma)) / q) * q``
* Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.

.. _hpo-space-support:

Search Space Types Supported by Each Tuner
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 
     - choice
     - choice(nested)
     - randint
     - uniform
     - quniform
     - loguniform
     - qloguniform
     - normal
     - qnormal
     - lognormal
     - qlognormal

   * - :class:`TPE <nni.algorithms.hpo.tpe_tuner.TpeTuner>`
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

   * - :class:`Random <nni.algorithms.hpo.random_tuner.RandomTuner>`
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

   * - :class:`Grid Search <nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner>`
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

   * - :class:`Anneal <nni.algorithms.hpo.hyperopt_tuner.HyperoptTuner>`
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

   * - :class:`Evolution <nni.algorithms.hpo.evolution_tuner.EvolutionTuner>`
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

   * - :class:`SMAC <nni.algorithms.hpo.smac_tuner.SMACTuner>`
     - ✓
     - 
     - ✓
     - ✓
     - ✓
     - ✓
     - 
     - 
     - 
     - 
     - 

   * - :class:`Batch <nni.algorithms.hpo.batch_tuner.BatchTuner>`
     - ✓
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 

   * - :class:`Hyperband <nni.algorithms.hpo.hyperband_advisor.Hyperband>`
     - ✓
     - 
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

   * - :class:`Metis <nni.algorithms.hpo.metis_tuner.MetisTuner>`
     - ✓
     - 
     - ✓
     - ✓
     - ✓
     - 
     - 
     - 
     - 
     - 
     - 

   * - :class:`BOHB <nni.algorithms.hpo.bohb_advisor.BOHB>`
     - ✓
     -
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

   * - :class:`GP <nni.algorithms.hpo.gp_tuner.GPTuner>`
     - ✓
     - 
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - 
     - 
     - 
     - 

   * - :class:`PBT <nni.algorithms.hpo.pbt_tuner.PBTTuner>`
     - ✓
     -
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

   * - :class:`DNGO <nni.algorithms.hpo.dngo_tuner.DNGOTuner>`
     - ✓
     - 
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - 
     - 
     - 
     - 


Known Limitations:

* GP Tuner, Metis Tuner and DNGO tuner support only **numerical values** in search space
  (``choice`` type values can be no-numerical with other tuners, e.g. string values).
  Both GP Tuner and Metis Tuner use Gaussian Process Regressor(GPR).
  GPR make predictions based on a kernel function and the 'distance' between different points,
  it's hard to get the true distance between no-numerical values.

* Note that for nested search space:

  * Only TPE/Random/Grid Search/Anneal/Evolution tuners support nested search space.
