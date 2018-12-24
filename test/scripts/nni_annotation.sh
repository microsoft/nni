#!/bin/bash

## ------Run annotation test------
echo ""
echo "===========================Testing: nni_annotation==========================="
cd ../../tools/
python3 -m unittest -v nni_annotation/test_annotation.py