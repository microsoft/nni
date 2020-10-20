# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

os.environ['NNI_PLATFORM'] = 'unittest'
os.environ['NNI_TRIAL_JOB_ID'] = 'test_trial_job_id'
os.environ["NNI_OUTPUT_DIR"] = "./unittest"
os.environ["NNI_SYS_DIR"] = "./unittest"
os.environ["NNI_EXP_ID"] = "test_exp_id"
os.environ["MULTI_PHASE"] = "true"
