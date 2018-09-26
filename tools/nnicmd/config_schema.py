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

import os
from schema import Schema, And, Use, Optional, Regex, Or

CONFIG_SCHEMA = Schema({
'authorName': str,
'experimentName': str,
'trialConcurrency': And(int, lambda n: 1 <=n <= 999999),
'maxExecDuration': Regex(r'^[1-9][0-9]*[s|m|h|d]$'),
'maxTrialNum': And(int, lambda x: 1 <= x <= 99999),
'trainingServicePlatform': And(str, lambda x: x in ['remote', 'local', 'pai']),
Optional('searchSpacePath'): os.path.exists,
'useAnnotation': bool,
'tuner': Or({
    'builtinTunerName': Or('TPE', 'Random', 'Anneal', 'Evolution'),
    'classArgs': {
        'optimize_mode': Or('maximize', 'minimize'),
        Optional('speed'): int
        },
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999),
},{
    'codeDir': os.path.exists,
    'classFileName': str,
    'className': str,
    Optional('classArgs'): dict,
    Optional('gpuNum'): And(int, lambda x: 0 <= x <= 99999),
}),
'trial':{
    'command': str,
    'codeDir': os.path.exists,
    'gpuNum': And(int, lambda x: 0 <= x <= 99999),
    Optional('cpuNum'): And(int, lambda x: 0 <= x <= 99999),
    Optional('memoryMB'): int,
    Optional('image'): str,
    Optional('dataDir'): str,
    Optional('outputDir'): str
    },
Optional('assessor'): Or({
    'builtinAssessorName': lambda x: x in ['Medianstop'],
    'classArgs': {
        'optimize_mode': lambda x: x in ['maximize', 'minimize']},
    'gpuNum': And(int, lambda x: 0 <= x <= 99999)
},{
    'codeDir': os.path.exists,
    'classFileName': str,
    'className': str,
    'classArgs': {
        'optimize_mode': lambda x: x in ['maximize', 'minimize']},
    'gpuNum': And(int, lambda x: 0 <= x <= 99999),
}),
Optional('machineList'):[Or({
    'ip': str,
    'port': And(int, lambda x: 0 < x < 65535),
    'username': str,
    'passwd': str
    },{
    'ip': str,
    'port': And(int, lambda x: 0 < x < 65535),
    'username': str,
    'sshKeyPath': os.path.exists,
    Optional('passphrase'): str
})],
Optional('paiConfig'):{
  'userName': str,
  'passWord': str,
  'host': str
}
})