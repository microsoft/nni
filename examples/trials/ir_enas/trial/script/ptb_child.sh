#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/ptb/nni_child_ptb.py \
  --search_for="enas" \
  --noreset_output_dir \
  --data_path="data/ptb/ptb.pkl" \
  --output_dir="outputs" \
  --batch_size=20 \
  --child_bptt_steps=35 \
  --num_epochs=100 \
  --child_rhn_depth=12 \
  --child_num_layers=1 \
  --child_lstm_hidden_size=720 \
  --child_lstm_e_keep=0.75 \
  --child_lstm_x_keep=0.25 \
  --child_lstm_h_keep=0.75 \
  --child_lstm_o_keep=0.25 \
  --nochild_lstm_e_skip \
  --child_grad_bound=10.0 \
  --child_lr=0.25 \
  --child_lr_dec_start=12 \
  --child_lr_dec_every=1 \
  --child_lr_dec_rate=0.95 \
  --child_lr_dec_min=0.0005 \
  --child_optim_algo="sgd" \
  --child_l2_reg=1e-7 \
  --child_steps=1327 \
  --log_every=50 \
  --controller_train_every=1 \
  --controller_train_steps=100 \
  --controller_num_aggregate=10 \
  --eval_every_epochs=1

