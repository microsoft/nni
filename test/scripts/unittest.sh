#!/bin/bash
set -e
CWD=${PWD}

# -------------For python unittest-------------

## ------Run annotation test------
echo ""
echo "===========================Testing: nni_annotation==========================="
#cd ${CWD}/../tools/
#python3 -m unittest -v nni_annotation/test_annotation.py

## Export certain environment variables for unittest code to work
export NNI_TRIAL_JOB_ID=test_trial_job_id
export NNI_PLATFORM=unittest

## ------Run sdk test------
echo ""
echo "===========================Testing: nni_sdk==========================="
#cd ${CWD}/../src/sdk/pynni/
#python3 -m unittest discover -v tests

# -------------For typescript unittest-------------
#cd ${CWD}/../ts/nni_manager
echo ""
echo "===========================Testing: nni_manager==========================="
#npm run test

# -------------For NASUI unittest-------------
#cd ${CWD}/../ts/nasui
echo ""
echo "===========================Testing: nasui==========================="
#CI=true npm test

## ------Run nnictl unit test------
echo ""
echo "===========================Testing: nnictl==========================="
#cd ${CWD}/../tools/nni_cmd/
#python3 -m unittest discover -v tests
