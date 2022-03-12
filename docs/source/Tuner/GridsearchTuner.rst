Grid Search Tuner
=================

Grid Search performs an exhaustive search through a search space.

For uniform and normal distributed parameters, grid search tuner samples them at progressively decreased intervals.

Usage
-----

Grid search tuner has no argument.

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   tuner:
     name: GridSearch
