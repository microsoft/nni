#!/bin/bash

CWD=${PWD}
# -------------For python unittest-------------
## Export certain environment variables for unittest code to work
export NNI_TRIAL_JOB_ID=test_trial_job_id
export NNI_PLATFORM=unittest

## ------Run sdk test------
echo "Testing: nni_sdk..."
cd ../src/sdk/pynni/
python3 -m unittest discover -v tests
cd ${CWD}

## ------Run annotation test------
echo "Testing: nni_annotation..."
cd ../tools/
python3 -m unittest -v nni_annotation/test_annotation.py
cd ${CWD}


# -------------For typescrip unittest-------------
cd ../src/nni_manager
echo "Testing: nni_manager..."
npm run tests
cd ${CWD}