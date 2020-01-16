NGPUS=4
SGPU=0
EGPU=$[NGPUS+SGPU-1]
GPU_ID=`seq -s , $SGPU $EGPU`
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=$NGPUS search.py \
    --dataset cifar10 --n_classes 10 --init_channels 16 --stem_multiplier 3 \
    --batch_size 64 --workers 1 --log_frequency 10 \
    --distributed --world_size $NGPUS --dist_url 'tcp://127.0.0.1:23343' \
    --regular_ratio 0.2 --regular_coeff 5 \
    --loss_alpha 1 --loss_T 2 \
    --w_lr 0.2 --alpha_lr 3e-4 --nasnet_lr 0.2 \
    --w_weight_decay 0. --alpha_weight_decay 0. \
    --share_module --interactive_type kl \
    --warmup_epochs 2 --epochs 32
