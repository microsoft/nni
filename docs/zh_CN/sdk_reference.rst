###########################
Python API 参考
###########################

Trial（尝试）
------------------------
..  autofunction:: nni.get_next_parameter
..  autofunction:: nni.get_current_parameter
..  autofunction:: nni.report_intermediate_result
..  autofunction:: nni.report_final_result
..  autofunction:: nni.get_sequence_id


Tuner（调参器）
------------------------
..  autoclass:: nni.tuner.Tuner
    :members:

..  autoclass:: nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner
    :members:

..  autoclass:: nni.evolution_tuner.evolution_tuner.EvolutionTuner
    :members:

..  autoclass:: nni.smac_tuner.smac_tuner.SMACTuner
    :members:

..  autoclass:: nni.gridsearch_tuner.gridsearch_tuner.GridSearchTuner
    :members:

..  autoclass:: nni.networkmorphism_tuner.networkmorphism_tuner.NetworkMorphismTuner
    :members:

..  autoclass:: nni.metis_tuner.metis_tuner.MetisTuner
    :members:

Assessor（评估器）
------------------------
..  autoclass:: nni.assessor.Assessor
    :members:

..  autoclass:: nni.curvefitting_assessor.curvefitting_assessor.CurvefittingAssessor
    :members:

..  autoclass:: nni.medianstop_assessor.medianstop_assessor.MedianstopAssessor
    :members:


Advisor
------------------------
..  autoclass:: nni.hyperband_advisor.hyperband_advisor.Hyperband
    :members:

..  autoclass:: nni.bohb_advisor.bohb_advisor.BOHB
    :members: