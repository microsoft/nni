# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit test of NNI Python modules.

Test cases of each module should be placed at same path of their source files.
For example if `nni/tool/annotation` has one test case, it should be placed at `test/ut/tool/test_annotation.py`;
if it has multiple test cases, they should be placed in `test/ut/tool/annotation/` directory.

"Legacy" test cases carried from NNI v1.x might not follow above convention:

  + Directory `sdk` contains old test cases previously in `src/sdk/pynni/tests`.
  + Directory `tools/nnictl` contains old test cases previously in `tools/nni_cmd/tests`.
  + Directory `tools/annotation` contains old test cases previously in `tools/nni_annotation` (removed).
  + Directory `tools/trial_tool` contains old test cases previously in `tools/nni_trial_tool/test`.
"""

import os

os.environ['NNI_PLATFORM'] = 'unittest'
os.environ['NNI_TRIAL_JOB_ID'] = 'test_trial_job_id'
os.environ['NNI_EXP_ID'] = 'test_experiment'
