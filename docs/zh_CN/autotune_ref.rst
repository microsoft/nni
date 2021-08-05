自动调优的 Python API 参考
=================================

.. contents::

Trial
-----

..  autofunction:: nni.get_next_parameter
..  autofunction:: nni.get_current_parameter
..  autofunction:: nni.report_intermediate_result
..  autofunction:: nni.report_final_result
..  autofunction:: nni.get_experiment_id
..  autofunction:: nni.get_trial_id
..  autofunction:: nni.get_sequence_id

Tuner
-----

..  autoclass:: nni.tuner.Tuner
    :members:

..  autoclass:: nni.algorithms.hpo.hyperopt_tuner.HyperoptTuner
    :members:

..  autoclass:: nni.algorithms.hpo.evolution_tuner.EvolutionTuner
    :members:

..  autoclass:: nni.algorithms.hpo.smac_tuner.SMACTuner
    :members:

..  autoclass:: nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner
    :members:

..  autoclass:: nni.algorithms.hpo.networkmorphism_tuner.NetworkMorphismTuner
    :members:

..  autoclass:: nni.algorithms.hpo.metis_tuner.MetisTuner
    :members:

..  autoclass:: nni.algorithms.hpo.ppo_tuner.PPOTuner
    :members:

..  autoclass:: nni.algorithms.hpo.batch_tuner.BatchTuner
    :members:

..  autoclass:: nni.algorithms.hpo.gp_tuner.GPTuner
    :members:

Assessor
--------

..  autoclass:: nni.assessor.Assessor
    :members:

..  autoclass:: nni.assessor.AssessResult
    :members:

..  autoclass:: nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor
    :members:

..  autoclass:: nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor
    :members:

Advisor
-------

..  autoclass:: nni.runtime.msg_dispatcher_base.MsgDispatcherBase
    :members:

..  autoclass:: nni.algorithms.hpo.hyperband_advisor.Hyperband
    :members:

..  autoclass:: nni.algorithms.hpo.bohb_advisor.BOHB
    :members:

Utilities
---------

..  autofunction:: nni.utils.merge_parameter
