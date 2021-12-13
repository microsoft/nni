Naive Evolution Tuners on NNI
=============================


Introduction
------------

Naive Evolution comes from `Large-Scale Evolution of Image Classifiers <https://arxiv.org/pdf/1703.01041.pdf>`__. It randomly initializes a population based on the search space. For each generation, it chooses better ones and does some mutation (e.g., changes a hyperparameter, adds/removes one layer, etc.) on them to get the next generation. Naive Evolution requires many trials to works but it's very simple and it's easily expanded with new features.

Usage
-----

classArgs Requirements
^^^^^^^^^^^^^^^^^^^^^^

* 
  **optimize_mode** (*maximize or minimize, optional, default = maximize*\ ) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.

* 
  **population_size** (*int value (should > 0), optional, default = 20*\ ) - the initial size of the population (trial num) in the evolution tuner. It's suggested that ``population_size`` be much larger than ``concurrency`` so users can get the most out of the algorithm (and at least ``concurrency``\ , or the tuner will fail on its first generation of parameters).

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # config.yml
   tuner:
     name: Evolution
     classArgs:
       optimize_mode: maximize
       population_size: 100

