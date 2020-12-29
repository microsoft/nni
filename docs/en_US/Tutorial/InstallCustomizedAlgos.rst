
**How to register customized algorithms as builtin tuners, assessors and advisors**
=======================================================================================

Overview
--------

NNI provides a lot of `builtin tuners <../Tuner/BuiltinTuner.rst>`_, `advisors <../Tuner/HyperbandAdvisor.rst>`__ and `assessors <../Assessor/BuiltinAssessor.rst>`__ can be used directly for Hyper Parameter Optimization, and some extra algorithms can be registered via ``nnictl algo register --meta <path_to_meta_file>`` after NNI is installed. You can check builtin algorithms via ``nnictl algo list`` command.

NNI also provides the ability to build your own customized tuners, advisors and assessors. To use the customized algorithm, users can simply follow the spec in experiment config file to properly reference the algorithm, which has been illustrated in the tutorials of `customized tuners <../Tuner/CustomizeTuner.rst>`_ / `advisors <../Tuner/CustomizeAdvisor.rst>`__ / `assessors <../Assessor/CustomizeAssessor.rst>`__.

NNI also allows users to install the customized algorithm as a builtin algorithm, in order for users to use the algorithm in the same way as NNI builtin tuners/advisors/assessors. More importantly, it becomes much easier for users to share or distribute their implemented algorithm to others. Customized tuners/advisors/assessors can be installed into NNI as builtin algorithms, once they are installed into NNI, you can use your customized algorithms the same way as builtin tuners/advisors/assessors in your experiment configuration file. For example, you built a customized tuner and installed it into NNI using a builtin name ``mytuner``, then you can use this tuner in your configuration file like below:

.. code-block:: yaml

   tuner:
     builtinTunerName: mytuner

Register customized algorithms as builtin tuners, assessors and advisors
------------------------------------------------------------------------

You can follow below steps to build a customized tuner/assessor/advisor, and register it into NNI as builtin algorithm.

1. Create a customized tuner/assessor/advisor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reference following instructions to create:


* `customized tuner <../Tuner/CustomizeTuner.rst>`_
* `customized assessor <../Assessor/CustomizeAssessor.rst>`_
* `customized advisor <../Tuner/CustomizeAdvisor.rst>`_

2. (Optional) Create a validator to validate classArgs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NNI provides a ``ClassArgsValidator`` interface for customized algorithms author to validate the classArgs parameters in experiment configuration file which are passed to customized algorithms constructors.
The ``ClassArgsValidator`` interface is defined as:

.. code-block:: python

   class ClassArgsValidator(object):
       def validate_class_args(self, **kwargs):
           """
           The classArgs fields in experiment configuration are packed as a dict and
           passed to validator as kwargs.
           """
           pass

For example, you can implement your validator such as:

.. code-block:: python

   from schema import Schema, Optional
   from nni import ClassArgsValidator

   class MedianstopClassArgsValidator(ClassArgsValidator):
       def validate_class_args(self, **kwargs):
           Schema({
               Optional('optimize_mode'): self.choices('optimize_mode', 'maximize', 'minimize'),
               Optional('start_step'): self.range('start_step', int, 0, 9999),
           }).validate(kwargs)

The validator will be invoked before experiment is started to check whether the classArgs fields are valid for your customized algorithms.

3. Install your customized algorithms into python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Firstly, the customized algorithms need to be prepared as a python package. Then you can install the package into python environment via:


* Run command ``python setup.py develop`` from the package directory, this command will install the package in development mode, this is recommended if your algorithm is under development.
* Run command ``python setup.py bdist_wheel`` from the package directory, this command build a whl file which is a pip installation source. Then run ``pip install <wheel file>`` to install it.

4. Prepare meta file
^^^^^^^^^^^^^^^^^^^^

Create a yaml file with following keys as meta file:


* ``algoType``: type of algorithms, could be one of ``tuner``, ``assessor``, ``advisor``
* ``builtinName``: builtin name used in experiment configuration file
* `className`: tuner class name, including its module name, for example: ``demo_tuner.DemoTuner``
* `classArgsValidator`: class args validator class name, including its module name, for example: ``demo_tuner.MyClassArgsValidator``

Following is an example of the yaml file:

.. code-block:: yaml

   algoType: tuner
   builtinName: demotuner
   className: demo_tuner.DemoTuner
   classArgsValidator: demo_tuner.MyClassArgsValidator

5. Register customized algorithms into NNI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run following command to register the customized algorithms as builtin algorithms in NNI:

.. code-block:: bash

   nnictl algo register --meta <path_to_meta_file>

The ``<path_to_meta_file>`` is the path to the yaml file your created in above section.

Reference `customized tuner example <../Tuner/InstallCustomizedTuner.rst>`_ for a full example.

6. Use the installed builtin algorithms in experiment
-----------------------------------------------------

Once your customized algorithms is installed, you can use it in experiment configuration file the same way as other builtin tuners/assessors/advisors, for example:

.. code-block:: yaml

   tuner:
     builtinTunerName: demotuner
     classArgs:
       #choice: maximize, minimize
       optimize_mode: maximize

Manage builtin algorithms using ``nnictl algo``
---------------------------------------------------

List builtin algorithms
^^^^^^^^^^^^^^^^^^^^^^^

Run following command to list the registered builtin algorithms:

.. code-block:: bash

   nnictl algo list
   +-----------------+------------+-----------+--------=-------------+------------------------------------------+
   |      Name       |    Type    | Source    |      Class Name      |               Module Name                |
   +-----------------+------------+-----------+----------------------+------------------------------------------+
   | TPE             | tuners     | nni       | HyperoptTuner        | nni.hyperopt_tuner.hyperopt_tuner        |
   | Random          | tuners     | nni       | HyperoptTuner        | nni.hyperopt_tuner.hyperopt_tuner        |
   | Anneal          | tuners     | nni       | HyperoptTuner        | nni.hyperopt_tuner.hyperopt_tuner        |
   | Evolution       | tuners     | nni       | EvolutionTuner       | nni.evolution_tuner.evolution_tuner      |
   | BatchTuner      | tuners     | nni       | BatchTuner           | nni.batch_tuner.batch_tuner              |
   | GridSearch      | tuners     | nni       | GridSearchTuner      | nni.gridsearch_tuner.gridsearch_tuner    |
   | NetworkMorphism | tuners     | nni       | NetworkMorphismTuner | nni.networkmorphism_tuner.networkmo...   |
   | MetisTuner      | tuners     | nni       | MetisTuner           | nni.metis_tuner.metis_tuner              |
   | GPTuner         | tuners     | nni       | GPTuner              | nni.gp_tuner.gp_tuner                    |
   | PBTTuner        | tuners     | nni       | PBTTuner             | nni.pbt_tuner.pbt_tuner                  |
   | SMAC            | tuners     | nni       | SMACTuner            | nni.smac_tuner.smac_tuner                |
   | PPOTuner        | tuners     | nni       | PPOTuner             | nni.ppo_tuner.ppo_tuner                  |
   | Medianstop      | assessors  | nni       | MedianstopAssessor   | nni.medianstop_assessor.medianstop_...   |
   | Curvefitting    | assessors  | nni       | CurvefittingAssessor | nni.curvefitting_assessor.curvefitt...   |
   | Hyperband       | advisors   | nni       | Hyperband            | nni.hyperband_advisor.hyperband_adv...   |
   | BOHB            | advisors   | nni       | BOHB                 | nni.bohb_advisor.bohb_advisor            |
   +-----------------+------------+-----------+----------------------+------------------------------------------+

Unregister builtin algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run following command to uninstall an installed package:

``nnictl algo unregister <builtin name>``

For example:

``nnictl algo unregister demotuner``
