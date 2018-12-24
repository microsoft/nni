#!/bin/bash

## Export certain environment variables for unittest code to work
export NNI_TRIAL_JOB_ID=test_trial_job_id
export NNI_PLATFORM=unittest

## ------Run sdk test------
echo "Testing: nni_sdk..."
cd ../../src/sdk/pynni/
python3 -m unittest discover -v tests
cd ${CWD}