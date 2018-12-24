#!/bin/bash
cd scripts
# -------------For python unittest-------------

## ------Run sdk test------
chmod +x nni_sdk.sh
./nni_sdk.sh

## ------Run annotation test------
chmod +x nni_annotation.sh
./nni_annotation.sh

# -------------For typescrip unittest-------------
chmod +x nni_manager.sh
./nni_manager.sh
