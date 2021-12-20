Batch Tuner
===========

Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type ``choice`` in the `search space spec <../Tutorial/SearchSpaceSpec.rst>`__.

Suggested scenario: If the configurations you want to try have been decided, you can list them in the SearchSpace file (using ``choice``) and run them using the batch tuner.

Usage
-----

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # config.yml
   tuner:
     name: BatchTuner

Note that the search space for BatchTuner should look like:

.. code-block:: json

   {
       "combine_params":
       {
           "_type" : "choice",
           "_value" : [{"optimizer": "Adam", "learning_rate": 0.00001},
                       {"optimizer": "Adam", "learning_rate": 0.0001},
                       {"optimizer": "Adam", "learning_rate": 0.001},
                       {"optimizer": "SGD", "learning_rate": 0.01},
                       {"optimizer": "SGD", "learning_rate": 0.005},
                       {"optimizer": "SGD", "learning_rate": 0.0002}]
       }
   }

The search space file should include the high-level key ``combine_params``. The type of params in the search space must be ``choice`` and the ``values`` must include all the combined params values.
