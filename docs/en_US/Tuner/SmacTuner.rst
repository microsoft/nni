SMAC Tuner
==========

`SMAC <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO in order to handle categorical parameters. The SMAC supported by nni is a wrapper on `the SMAC3 github repo <https://github.com/automl/SMAC3>`__.

Note that SMAC on nni only supports a subset of the types in the `search space spec <../Tutorial/SearchSpaceSpec.rst>`__: ``choice``, ``randint``, ``uniform``, ``loguniform``, and ``quniform``.

Usage
-----

Installation
^^^^^^^^^^^^

SMAC has dependencies that need to be installed by following command before the first usage. As a reminder, ``swig`` is required for SMAC: for Ubuntu ``swig`` can be installed with ``apt``.

.. code-block:: bash

   pip install nni[SMAC]

classArgs requirements
^^^^^^^^^^^^^^^^^^^^^^

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **config_dedup** (*True or False, optional, default = False*) - If True, the tuner will not generate a configuration that has been already generated. If False, a configuration may be generated twice, but it is rare for a relatively large search space.

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # config.yml
   tuner:
     name: SMAC
     classArgs:
       optimize_mode: maximize
