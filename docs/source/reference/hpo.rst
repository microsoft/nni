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

.. autoclass:: nni.algorithms.hpo.batch_tuner.BatchTuner
.. autoclass:: nni.algorithms.hpo.bohb_advisor.BOHB
.. autoclass:: nni.algorithms.hpo.dngo_tuner.DNGOTuner
.. autoclass:: nni.algorithms.hpo.evolution_tuner.EvolutionTuner
.. autoclass:: nni.algorithms.hpo.gp_tuner.GPTuner
.. autoclass:: nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner
.. autoclass:: nni.algorithms.hpo.hyperband_advisor.Hyperband
.. autoclass:: nni.algorithms.hpo.hyperopt_tuner.HyperoptTuner
.. autoclass:: nni.algorithms.hpo.metis_tuner.MetisTuner
.. autoclass:: nni.algorithms.hpo.pbt_tuner.PBTTuner
.. autoclass:: nni.algorithms.hpo.ppo_tuner.PPOTuner
.. autoclass:: nni.algorithms.hpo.random_tuner.RandomTuner
.. autoclass:: nni.algorithms.hpo.smac_tuner.SMACTuner
.. autoclass:: nni.algorithms.hpo.tpe_tuner.TpeTuner
.. autoclass:: nni.algorithms.hpo.tpe_tuner.TpeArguments

Assessors
---------

.. autoclass:: nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor
.. autoclass:: nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor

Customization
-------------

.. autoclass:: nni.assessor.AssessResult
    :members:
.. autoclass:: nni.assessor.Assessor
    :members:
.. autoclass:: nni.tuner.Tuner
    :members:
