# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


ModuleName = {
    'TPE': 'nni.hyperopt_tuner.hyperopt_tuner',
    'Random': 'nni.hyperopt_tuner.hyperopt_tuner',
    'Anneal': 'nni.hyperopt_tuner.hyperopt_tuner',
    'Evolution': 'nni.evolution_tuner.evolution_tuner',
    'SMAC': 'nni.smac_tuner.smac_tuner',
    'BatchTuner': 'nni.batch_tuner.batch_tuner',
    'Medianstop': 'nni.medianstop_assessor.medianstop_assessor',
    'GridSearch': 'nni.gridsearch_tuner.gridsearch_tuner',
    'NetworkMorphism': 'nni.networkmorphism_tuner.networkmorphism_tuner',
    'Curvefitting': 'nni.curvefitting_assessor.curvefitting_assessor',
    'MetisTuner': 'nni.metis_tuner.metis_tuner',
    'GPTuner': 'nni.gp_tuner.gp_tuner',
    'PPOTuner': 'nni.ppo_tuner.ppo_tuner'
}

ClassName = {
    'TPE': 'HyperoptTuner',
    'Random': 'HyperoptTuner',
    'Anneal': 'HyperoptTuner',
    'Evolution': 'EvolutionTuner',
    'SMAC': 'SMACTuner',
    'BatchTuner': 'BatchTuner',
    'GridSearch': 'GridSearchTuner',
    'NetworkMorphism':'NetworkMorphismTuner',
    'MetisTuner':'MetisTuner',
    'GPTuner':'GPTuner',
    'PPOTuner': 'PPOTuner',

    'Medianstop': 'MedianstopAssessor',
    'Curvefitting': 'CurvefittingAssessor'
}

ClassArgs = {
    'TPE': {
        'algorithm_name': 'tpe'
    },
    'Random': {
        'algorithm_name': 'random_search'
    },
    'Anneal': {
        'algorithm_name': 'anneal'
    }
}

AdvisorModuleName = {
    'Hyperband': 'nni.hyperband_advisor.hyperband_advisor',
    'BOHB': 'nni.bohb_advisor.bohb_advisor'
}

AdvisorClassName = {
    'Hyperband': 'Hyperband',
    'BOHB': 'BOHB'
}
