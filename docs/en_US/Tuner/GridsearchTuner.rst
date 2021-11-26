Grid Search on NNI
==================

Grid Search
-----------

1. Introduction
---------------

Grid Search performs an exhaustive search through a search space.

For uniform and normal distributed parameters, grid search tuner samples them at progressively decreased intervals.

2. Usage
--------

Grid search tuner has no argument.

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   tuner:
     name: GridSearch
