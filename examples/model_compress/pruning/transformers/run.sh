#!/bin/bash

# Usage: ./run.sh script gpu_id task

export CUDA_VISIBLE_DEVICES=$2
SOURCE_CODE=$1
TASK_NAME=$3
PRETRAINED_MODEL='bert-base-uncased'          # 'distilbert-base-uncased'  'roberta-base'   'bert-base-cased'   'bert-base-uncased'
MAX_LENGTH=128
BATCH_SIZE=32
LR=2e-5
N_EPOCHS=3
SEED=2021

time=$(date "+%Y%m%d%H%M%S")
OUTDIR="models_${PRETRAINED_MODEL}_${SOURCE_CODE}_${TASK_NAME}_$time/"

TASK_LIST=('cola' 'sst2' 'mrpc' 'stsb' 'qqp' 'mnli' 'qnli' 'rte' 'wnli')
if [[ ${TASK_LIST[*]} =~ (^|[[:space:]])$TASK_NAME($|[[:space:]]) ]]; then
    mkdir $OUTDIR
    python $SOURCE_CODE \
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
