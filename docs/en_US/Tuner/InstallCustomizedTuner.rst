How to register a customized tuner as a builtin tuner
=====================================================

You can following below steps to register a customized tuner in ``nni/examples/tuners/customized_tuner`` as a builtin tuner.

Install the customized tuner package into python environment
------------------------------------------------------------

There are 2 options to install the package into python environment:

Option 1: install from directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From ``nni/examples/tuners/customized_tuner`` directory, run:

``python setup.py develop``

This command will build the ``nni/examples/tuners/customized_tuner`` directory as a pip installation source.

Option 2: install from whl file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Step 1: From ``nni/examples/tuners/customized_tuner`` directory, run:

``python setup.py bdist_wheel``

This command build a whl file which is a pip installation source.

Step 2: Run command:

``pip install dist/demo_tuner-0.1-py3-none-any.whl``

Register the customized tuner as builtin tuner:
-----------------------------------------------

Run following command:

``nnictl algo register --meta meta_file.yml``

Check the registered builtin algorithms
---------------------------------------

Then run command ``nnictl algo list``\ , you should be able to see that demotuner is installed:

.. code-block:: bash

   +-----------------+------------+-----------+--------=-------------+------------------------------------------+
   |      Name       |    Type    |   source  |      Class Name      |               Module Name                |
   +-----------------+------------+-----------+----------------------+------------------------------------------+
   | demotuner       | tuners     |    User   | DemoTuner            | demo_tuner                               |
   +-----------------+------------+-----------+----------------------+------------------------------------------+

Use the installed tuner in experiment
-------------------------------------

Now you can use the demotuner in experiment configuration file the same way as other builtin tuners:

.. code-block:: yaml

   tuner:
     builtinTunerName: demotuner
     classArgs:
       #choice: maximize, minimize
       optimize_mode: maximize
