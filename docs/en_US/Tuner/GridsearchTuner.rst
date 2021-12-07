Grid Search on NNI
==================

Grid Search
-----------

1. Introduction
---------------

Grid Search performs an exhaustive search through a manually specified subset of the hyperparameter space defined in the searchspace file. 

Note that the only acceptable types within the search space are ``choice``\ , ``quniform``\ , and ``randint``.

2. Usage
--------

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # config.yml
   tuner:
     builtinTunerName: GridSearch