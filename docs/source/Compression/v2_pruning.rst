Pruning V2
==========

Pruning V2 is a refactoring of the old version and provides more powerful functions.
Compared with the old version, the iterative pruning process is detached from the pruner and the pruner is only responsible for pruning and generating the masks once.
What's more, pruning V2 unifies the pruning process and provides a more free combination of pruning components.
Task generator only cares about the pruning effect that should be achieved in each round, and uses a config list to express how to pruning in the next step.
Pruner will reset with the model and config list given by task generator then generate the masks in current step.

For a clearer structure vision, please refer to the figure below.

.. image:: ../../img/pruning_process.png
   :target: ../../img/pruning_process.png
   :alt:

In V2, a pruning process is usually driven by a pruning scheduler, it contains a specific pruner and a task generator.
But users can also use pruner directly like in the pruning V1.

For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Pruning Algorithms <v2_pruning_algo>
    Pruning Scheduler <v2_scheduler>
    Pruning Config List <v2_pruning_config_list>
