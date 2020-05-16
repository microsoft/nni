# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

BuiltinAlgorithms = {
    'tuners': [
        {
            'name': 'TPE',
            'class_name': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner',
            'class_args': {
                'algorithm_name': 'tpe'
            }
        },
        {
            'name': 'Random',
            'class_name': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner',
            'class_args': {
                'algorithm_name': 'random_search'
            }
        },
        {
            'name': 'Anneal',
            'class_name': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner',
            'class_args': {
                'algorithm_name': 'anneal'
            }
        },
        {
            'name': 'Evolution',
            'class_name': 'nni.evolution_tuner.evolution_tuner.EvolutionTuner'
        },
        {
            'name': 'BatchTuner',
            'class_name': 'nni.batch_tuner.batch_tuner.BatchTuner'
        },
        {
            'name': 'GridSearch',
            'class_name': 'nni.gridsearch_tuner.gridsearch_tuner.GridSearchTuner'
        },
        {
            'name': 'NetworkMorphism',
            'class_name': 'nni.networkmorphism_tuner.networkmorphism_tuner.NetworkMorphismTuner'
        },
        {
            'name': 'MetisTuner',
            'class_name': 'nni.metis_tuner.metis_tuner.MetisTuner'
        },
        {
            'name': 'GPTuner',
            'class_name': 'nni.gp_tuner.gp_tuner.GPTuner'
        },
        {
            'name': 'PBTTuner',
            'class_name': 'nni.pbt_tuner.pbt_tuner.PBTTuner'
        }
    ],
    'assessors': [
        {
            'name': 'Medianstop',
            'class_name': 'nni.medianstop_assessor.medianstop_assessor.MedianstopAssessor'
        },
        {
            'name': 'Curvefitting',
            'class_name': 'nni.curvefitting_assessor.curvefitting_assessor.CurvefittingAssessor'
        },
    ],
    'advisors': [
        {
            'name': 'Hyperband',
            'class_name': 'nni.hyperband_advisor.hyperband_advisor.Hyperband'
        }
    ]
}

ModuleName = {
    'TPE': 'nni.hyperopt_tuner.hyperopt_tuner',
    'Random': 'nni.hyperopt_tuner.hyperopt_tuner',
    'Anneal': 'nni.hyperopt_tuner.hyperopt_tuner',
    'Evolution': 'nni.evolution_tuner.evolution_tuner',
    #'SMAC': 'nni.smac_tuner.smac_tuner',
    'BatchTuner': 'nni.batch_tuner.batch_tuner',
    'Medianstop': 'nni.medianstop_assessor.medianstop_assessor',
    'GridSearch': 'nni.gridsearch_tuner.gridsearch_tuner',
    'NetworkMorphism': 'nni.networkmorphism_tuner.networkmorphism_tuner',
    'Curvefitting': 'nni.curvefitting_assessor.curvefitting_assessor',
    'MetisTuner': 'nni.metis_tuner.metis_tuner',
    'GPTuner': 'nni.gp_tuner.gp_tuner',
    'PBTTuner': 'nni.pbt_tuner.pbt_tuner'
}

ClassName = {
    'TPE': 'HyperoptTuner',
    'Random': 'HyperoptTuner',
    'Anneal': 'HyperoptTuner',
    'Evolution': 'EvolutionTuner',
    'BatchTuner': 'BatchTuner',
    'GridSearch': 'GridSearchTuner',
    'NetworkMorphism':'NetworkMorphismTuner',
    'MetisTuner':'MetisTuner',
    'GPTuner':'GPTuner',
    'PBTTuner': 'PBTTuner',
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
}

AdvisorClassName = {
    'Hyperband': 'Hyperband',
}
