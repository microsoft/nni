#!/bin/bash
set -e
CWD=${PWD}

# -------------For python unittest-------------

## ------Run annotation test------
echo ""
echo "===========================Testing: nni_annotation==========================="
cd ${CWD}/../tools/
python3 -m unittest -v nni_annotation/test_annotation.py

## Export certain environment variables for unittest code to work
export NNI_TRIAL_JOB_ID=test_trial_job_id
export NNI_PLATFORM=unittest

## ------Run sdk test------
echo ""
echo "===========================Testing: nni_sdk==========================="
cd ${CWD}/../src/sdk/pynni/
python3 -m pip install --user -r nni/smac_tuner/requirements.txt
python3 -m unittest discover -v tests

# -------------For typescript unittest-------------
cd ${CWD}/../src/nni_manager
echo ""
echo "===========================Testing: nni_manager==========================="
npm run test
