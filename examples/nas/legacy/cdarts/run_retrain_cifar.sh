NGPUS=4
SGPU=0
EGPU=$[NGPUS+SGPU-1]
GPU_ID=`seq -s , $SGPU $EGPU`
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=$NGPUS retrain.py \
    --dataset cifar10 --n_classes 10 --init_channels 36 --stem_multiplier 3 \
    --arc_checkpoint 'epoch_31.json' \
    --batch_size 128 --workers 1 --log_frequency 10 \
    --world_size $NGPUS --weight_decay 5e-4 \
    --distributed --dist_url 'tcp://127.0.0.1:26443' \
    --lr 0.1 --warmup_epochs 0 --epochs 600 \
    --cutout_length 16 --aux_weight 0.4 --drop_path_prob 0.3 \
    --label_smooth 0.0 --mixup_alpha 0
