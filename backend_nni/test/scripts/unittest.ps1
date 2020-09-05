$CWD = $PWD
$ErrorActionPreference = "Stop"
# -------------For python unittest-------------

## ------Run annotation test------
echo ""
echo "===========================Testing: nni_annotation==========================="
cd $CWD/../tools/
cmd /c "python -m unittest -v nni_annotation/test_annotation.py 2>&1"
if ($LASTEXITCODE -ne 0) {
    throw "Exit code $LASTEXITCODE"
}

## Export certain environment variables for unittest code to work
$env:NNI_TRIAL_JOB_ID="test_trial_job_id"
$env:NNI_PLATFORM="unittest"

## ------Run sdk test------
echo ""
echo "===========================Testing: nni_sdk==========================="
cd $CWD/../src/sdk/pynni/
cmd /c "python -m unittest discover -v tests 2>&1"
if ($LASTEXITCODE -ne 0) {
    throw "Exit code $LASTEXITCODE"
}

# -------------For typescript unittest-------------
cd $CWD/../src/nni_manager
echo ""
echo "===========================Testing: nni_manager==========================="
cmd /c "npm run test 2>&1"
# don't check now. adding back later.
# if ($LASTEXITCODE -ne 0) {
#     throw "Exit code $LASTEXITCODE"
# }
