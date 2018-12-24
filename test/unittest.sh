#!/bin/bash

cd scripts
# -------------For python unittest-------------

## ------Run sdk test------
./nni_sdk.sh

## ------Run annotation test------
./nni_annotation.sh

# -------------For typescrip unittest-------------
./nni_manager.sh
