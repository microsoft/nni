How to install customized tuner as a builtin tuner
==================================================

You can following below steps to install a customized tuner in ``nni/examples/tuners/customized_tuner`` as a builtin tuner.

Prepare installation source and install package
-----------------------------------------------

There are 2 options to install this customized tuner:

Option 1: install from directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Step 1: From ``nni/examples/tuners/customized_tuner`` directory, run:

``python setup.py develop``

This command will build the ``nni/examples/tuners/customized_tuner`` directory as a pip installation source.

Step 2: Run command:

``nnictl package install ./``

Option 2: install from whl file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Step 1: From ``nni/examples/tuners/customized_tuner`` directory, run:

``python setup.py bdist_wheel``

This command build a whl file which is a pip installation source.

Step 2: Run command:

``nnictl package install dist/demo_tuner-0.1-py3-none-any.whl``

Check the installed package
---------------------------

Then run command ``nnictl package list``\ , you should be able to see that demotuner is installed:

.. code-block:: bash

   +-----------------+------------+-----------+--------=-------------+------------------------------------------+
   |      Name       |    Type    | Installed |      Class Name      |               Module Name                |
   +-----------------+------------+-----------+----------------------+------------------------------------------+
   | demotuner       | tuners     | Yes       | DemoTuner            | demo_tuner                               |
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
