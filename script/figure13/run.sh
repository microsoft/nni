#!/bin/bash
source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

python coarsegrained_propagate.py
python finegrained_propagate.py

python draw_finegrained.py
python draw_coarse.py