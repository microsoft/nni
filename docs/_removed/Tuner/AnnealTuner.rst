Anneal Tuner
============

This simple annealing algorithm begins by sampling from the prior but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on random search that leverages smoothness in the response surface. The annealing rate is not adaptive.

Usage
-----

classArgs Requirements
^^^^^^^^^^^^^^^^^^^^^^

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # config.yml
   tuner:
     name: Anneal
     classArgs:
       optimize_mode: maximize
