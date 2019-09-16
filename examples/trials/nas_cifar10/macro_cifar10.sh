#!/bin/bash
set -e
export PYTHONPATH="$(pwd)"

python3 src/cifar10/nni_child_cifar10.py \
  --data_format="NCHW" \
  --search_for="macro" \
  --reset_output_dir \
  --data_path="data/cifar10" \
  --output_dir="outputs" \
  --train_data_size=45000 \
  --batch_size=100 \
  --num_epochs=8 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_use_aux_heads \
  --child_num_layers=12 \
  --child_out_filters=36 \
  --child_l2_reg=0.0002 \
  --child_num_branches=6 \
  --child_num_cell_layers=5 \
  --child_keep_prob=0.50 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.001 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --child_mode="subgraph" \
  "$@"

