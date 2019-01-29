###########################
Python API Reference
###########################

API for trial code
------------------------
..  autofunction:: nni.get_next_parameter
..  autofunction:: nni.get_current_parameter
..  autofunction:: nni.report_intermediate_result
..  autofunction:: nni.report_final_result
..  autofunction:: nni.get_sequence_id


API for tuners
------------------------
..  autoclass:: nni.tuner.Tuner
    :members:

..  autoclass:: nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner
    :members:

..  autoclass:: nni.evolution_tuner.evolution_tuner.EvolutionTuner
    :members:

..  autoclass:: nni.gridsearch_tuner.gridsearch_tuner.GridSearchTuner
    :members:

..  autoclass:: nni.smac_tuner.smac_tuner.SMACTuner
    :members:

API for assessors
------------------------
..  autoclass:: nni.assessor.Assessor
    :members:

..  autoclass:: nni.curvefitting_assessor.curvefitting_assessor.CurvefittingAssessor
    :members:

..  autoclass:: nni.medianstop_assessor.medianstop_assessor.MedianstopAssessor
    :members:


API for Advisors
------------------------
..  autoclass:: nni.hyperband_advisor.hyperband_advisor.Hyperband