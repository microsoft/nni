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
            'accept_class_args': False,
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
            'class_name': 'nni.evolution_tuner.evolution_tuner.EvolutionTuner',
            'class_args_validator': 'nni.evolution_tuner.evolution_tuner.EvolutionClassArgsValidator'
        },
        {
            'name': 'BatchTuner',
            'class_name': 'nni.batch_tuner.batch_tuner.BatchTuner',
            'accept_class_args': False,
        },
        {
            'name': 'GridSearch',
            'class_name': 'nni.gridsearch_tuner.gridsearch_tuner.GridSearchTuner',
            'accept_class_args': False,
        },
        {
            'name': 'NetworkMorphism',
            'class_name': 'nni.networkmorphism_tuner.networkmorphism_tuner.NetworkMorphismTuner',
            'class_args_validator': 'nni.networkmorphism_tuner.networkmorphism_tuner.NetworkMorphismClassArgsValidator'
        },
        {
            'name': 'MetisTuner',
            'class_name': 'nni.metis_tuner.metis_tuner.MetisTuner',
            'class_args_validator': 'nni.metis_tuner.metis_tuner.MetisClassArgsValidator'
        },
        {
            'name': 'GPTuner',
            'class_name': 'nni.gp_tuner.gp_tuner.GPTuner',
            'class_args_validator': 'nni.gp_tuner.gp_tuner.GPClassArgsValidator'
        },
        {
            'name': 'PBTTuner',
            'class_name': 'nni.pbt_tuner.pbt_tuner.PBTTuner',
            'class_args_validator': 'nni.pbt_tuner.pbt_tuner.PBTClassArgsValidator'
        },
        {
            'name': 'RegularizedEvolutionTuner',
            'class_name': 'nni.regularized_evolution_tuner.regularized_evolution_tuner.RegularizedEvolutionTuner',
            'class_args_validator': 'nni.regularized_evolution_tuner.regularized_evolution_tuner.EvolutionClassArgsValidator'
        }
    ],
    'assessors': [
        {
            'name': 'Medianstop',
            'class_name': 'nni.medianstop_assessor.medianstop_assessor.MedianstopAssessor',
            'class_args_validator': 'nni.medianstop_assessor.medianstop_assessor.MedianstopClassArgsValidator'
        },
        {
            'name': 'Curvefitting',
            'class_name': 'nni.curvefitting_assessor.curvefitting_assessor.CurvefittingAssessor',
            'class_args_validator': 'nni.curvefitting_assessor.curvefitting_assessor.CurvefittingClassArgsValidator'
        },
    ],
    'advisors': [
        {
            'name': 'Hyperband',
            'class_name': 'nni.hyperband_advisor.hyperband_advisor.Hyperband',
            'class_args_validator': 'nni.hyperband_advisor.hyperband_advisor.HyperbandClassArgsValidator'
        }
    ]
}
