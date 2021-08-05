# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

API_ROOT_URL = '/api/v1/nni-pai'

BASE_URL = 'http://{}'

LOG_DIR = os.environ['NNI_OUTPUT_DIR']

NNI_PLATFORM = os.environ['NNI_PLATFORM']

STDOUT_FULL_PATH = os.path.join(LOG_DIR, 'stdout')

STDERR_FULL_PATH = os.path.join(LOG_DIR, 'stderr')

STDOUT_API = '/stdout'
VERSION_API = '/version'
PARAMETER_META_API = '/parameter-file-meta'
NNI_SYS_DIR = os.environ['NNI_SYS_DIR']
NNI_TRIAL_JOB_ID = os.environ['NNI_TRIAL_JOB_ID']
NNI_EXP_ID = os.environ['NNI_EXP_ID']
MULTI_PHASE = os.environ['MULTI_PHASE']
