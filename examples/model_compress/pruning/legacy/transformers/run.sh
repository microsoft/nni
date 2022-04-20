#!/bin/bash

# Usage: ./run.sh gpu_id glue_task

export CUDA_VISIBLE_DEVICES=$1
TASK_NAME=$2                                  # "cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"
PRETRAINED_MODEL="bert-base-uncased"          # "distilbert-base-uncased", "roberta-base", "bert-base-cased", ...

# parameters for pruning
SPARSITY=0.5
RANKING_CRITERION=l1_weight                   # "l1_weight", "l2_weight", "l1_activation", "l2_activation", "taylorfo"
NUM_ITERATIONS=1                              # 1 for one-shot pruning
EPOCHS_PER_ITERATION=1

# other training parameters, no need to change
MAX_LENGTH=128
BATCH_SIZE=32
LR=2e-5
N_EPOCHS=3

time=$(date "+%Y%m%d%H%M%S")
OUTDIR="models_${PRETRAINED_MODEL}_${TASK_NAME}_$time/"

TASK_LIST=("cola" "sst2" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte" "wnli")
if [[ ${TASK_LIST[*]} =~ (^|[[:space:]])$TASK_NAME($|[[:space:]]) ]]; then
    mkdir $OUTDIR
    python transformer_pruning.py \
	   --sparsity $SPARSITY \
	   --ranking_criterion $RANKING_CRITERION \
	   --num_iterations $NUM_ITERATIONS \
	   --epochs_per_iteration $EPOCHS_PER_ITERATION \
	   --speedup \
	   --model_name $PRETRAINED_MODEL \
	   --task_name $TASK_NAME \
	   --max_length $MAX_LENGTH \
	   --batch_size $BATCH_SIZE \
	   --learning_rate $LR \
	   --num_train_epochs $N_EPOCHS \
	   --output_dir $OUTDIR \
	   2>&1 | tee "$OUTDIR/output.log"
else
    echo "Unsupported task $TASK_NAME."
fi
