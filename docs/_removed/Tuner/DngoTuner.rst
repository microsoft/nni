DNGO Tuner
==========

Usage
-----

Installation
^^^^^^^^^^^^

classArgs requirements
^^^^^^^^^^^^^^^^^^^^^^

* **optimize_mode** (*'maximize' or 'minimize'*) - If 'maximize', the tuner will target to maximize metrics. If 'minimize', the tuner will target to minimize metrics.
* **sample_size** (*int, default = 1000*) - Number of samples to select in each iteration. The best one will be picked from the samples as the next trial.
* **trials_per_update** (*int, default = 20*) - Number of trials to collect before updating the model.
* **num_epochs_per_training** (*int, default = 500*) - Number of epochs to train DNGO model.

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # config.yml
   tuner:
     name: DNGOTuner
     classArgs:
       optimize_mode: maximize
