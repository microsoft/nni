#!/bin/bash

# Usage: ./run.sh gpu_id task

export CUDA_VISIBLE_DEVICES=$1
TASK_NAME=$2
PRETRAINED_MODEL='bert-base-uncased'          # example: 'distilbert-base-uncased'  'roberta-base'   'bert-base-cased'   'bert-base-uncased'

# parameters for pruning
USAGE=2                       # change to different numbers to run examples with different configs
SPARSITY=0.5
RANKING_CRITERION=l1_weight
NUM_ITERATIONS=1              # 1 for one-shot pruning
EPOCHS_PER_ITERATION=1

# other training parameters, no need to change
MAX_LENGTH=128
BATCH_SIZE=32
LR=2e-5
N_EPOCHS=3
SEED=2021

time=$(date "+%Y%m%d%H%M%S")
OUTDIR="models_${PRETRAINED_MODEL}_${TASK_NAME}_$time/"

TASK_LIST=('cola' 'sst2' 'mrpc' 'stsb' 'qqp' 'mnli' 'qnli' 'rte' 'wnli')
if [[ ${TASK_LIST[*]} =~ (^|[[:space:]])$TASK_NAME($|[[:space:]]) ]]; then
    mkdir $OUTDIR
    python transformer_pruning.py \
	   --sparsity $SPARSITY \
	   --ranking_criterion $RANKING_CRITERION \
	   --num_iterations $NUM_ITERATIONS \
	   --epochs_per_iteration $EPOCHS_PER_ITERATION \
	   --speed_up \
	   --seed $SEED \
	   --model_name_or_path $PRETRAINED_MODEL \
	   --task_name $TASK_NAME \
	   --max_length $MAX_LENGTH \
	   --per_device_train_batch_size $BATCH_SIZE \
	   --learning_rate $LR \
	   --num_train_epochs $N_EPOCHS \
	   --output_dir $OUTDIR \
	   2>&1 | tee "$OUTDIR/output.log"
else
    echo "Unsupported task $TASK_NAME."
fi
