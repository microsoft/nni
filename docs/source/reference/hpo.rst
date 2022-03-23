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
    :members:
.. autoclass:: nni.algorithms.hpo.bohb_advisor.BOHB
    :members:
.. autoclass:: nni.algorithms.hpo.dngo_tuner.DNGOTuner
    :members:
.. autoclass:: nni.algorithms.hpo.evolution_tuner.EvolutionTuner
    :members:
.. autoclass:: nni.algorithms.hpo.gp_tuner.GPTuner
    :members:
.. autoclass:: nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner
    :members:
.. autoclass:: nni.algorithms.hpo.hyperband_advisor.Hyperband
    :members:
.. autoclass:: nni.algorithms.hpo.hyperopt_tuner.HyperoptTuner
    :members:
.. autoclass:: nni.algorithms.hpo.metis_tuner.MetisTuner
    :members:
.. autoclass:: nni.algorithms.hpo.pbt_tuner.PBTTuner
    :members:
.. autoclass:: nni.algorithms.hpo.ppo_tuner.PPOTuner
    :members:
.. autoclass:: nni.algorithms.hpo.random_tuner.RandomTuner
    :members:
.. autoclass:: nni.algorithms.hpo.smac_tuner.SMACTuner
    :members:
.. autoclass:: nni.algorithms.hpo.tpe_tuner.TpeTuner
    :members:
.. autoclass:: nni.algorithms.hpo.tpe_tuner.TpeArguments

Assessors
---------

.. autoclass:: nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor
    :members:
.. autoclass:: nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor
    :members:

Customization
-------------

.. autoclass:: nni.assessor.AssessResult
    :members:
.. autoclass:: nni.assessor.Assessor
    :members:
.. autoclass:: nni.tuner.Tuner
    :members:
