# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

BuiltinAlgorithms = {
    'tuners': [
        {
            'name': 'TPE',
            'class_name': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner',
            'class_args': {
                'algorithm_name': 'tpe'
            },
            'class_args_validator': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptClassArgsValidator'
        },
        {
            'name': 'Random',
            'class_name': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner',
            'class_args': {
                'algorithm_name': 'random_search'
            },
            'class_args_validator': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptClassArgsValidator'
        },
        {
            'name': 'Anneal',
            'class_name': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner',
            'class_args': {
                'algorithm_name': 'anneal'
            },
            'class_args_validator': 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptClassArgsValidator'
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
