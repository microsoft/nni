#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/cifar10/nni_child_cifar10.py \
  --data_format="NCHW" \
  --search_for="micro" \
  --reset_output_dir \
  --data_path="data/cifar10" \
  --output_dir="outputs" \
  --train_data_size=45000 \
  --batch_size=160 \
  --num_epochs=150 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_use_aux_heads \
  --child_num_layers=6 \
  --child_out_filters=20 \
  --child_l2_reg=1e-4 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --child_keep_prob=0.90 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.0005 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --controller_train_every=1 \
  --controller_num_aggregate=10 \
  --controller_train_steps=30 \
  "$@"

