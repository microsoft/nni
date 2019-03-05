# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
    'MetisTuner': 'nni.metis_tuner.metis_tuner'
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