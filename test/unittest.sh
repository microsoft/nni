#!/bin/bash
CWD=${PWD}
cd scripts
# -------------For python unittest-------------

## ------Run sdk test------
source nni_sdk.sh

## ------Run annotation test------
cd ${CWD}
source nni_annotation.sh

# -------------For typescrip unittest-------------
cd ${CWD}
source nni_manager.sh
