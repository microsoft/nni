HPO API Reference
=================

Trial APIs
----------

.. autofunction:: nni.get_experiment_id
.. autofunction:: nni.get_next_parameter
.. autofunction:: nni.get_sequence_id
.. autofunction:: nni.get_trial_id
.. autofunction:: nni.report_final_result
.. autofunction:: nni.report_intermediate_result

Tuners
------

Batch Tuner
^^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.batch_tuner.BatchTuner

BOHB Tuner
^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.bohb_advisor.BOHB

DNGO Tuner
^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.dngo_tuner.DNGOTuner

Evolution Tuner
^^^^^^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.evolution_tuner.EvolutionTuner

GP Tuner
^^^^^^^^
.. autoclass:: nni.algorithms.hpo.gp_tuner.GPTuner

Grid Search Tuner
^^^^^^^^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner

Hyperband Tuner
^^^^^^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.hyperband_advisor.Hyperband

Hyperopt Tuner
^^^^^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.hyperopt_tuner.HyperoptTuner

Metis Tuner
^^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.metis_tuner.MetisTuner

PBT Tuner
^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.pbt_tuner.PBTTuner

PPO Tuner
^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.ppo_tuner.PPOTuner

Random Tuner
^^^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.random_tuner.RandomTuner

SMAC Tuner
^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.smac_tuner.SMACTuner

TPE Tuner
^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.tpe_tuner.TpeTuner
.. autoclass:: nni.algorithms.hpo.tpe_tuner.TpeArguments

Assessors
---------

Curve Fitting Assessor
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor

Median Stop Assessor
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor

Customization
-------------

.. autoclass:: nni.assessor.AssessResult
    :members:
.. autoclass:: nni.assessor.Assessor
    :members:
.. autoclass:: nni.tuner.Tuner
    :members:
