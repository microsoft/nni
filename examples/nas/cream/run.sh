CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./examples/nas/cream/distributed_train.sh 8 \
    --data ./data/imagenet/ --sched spos_linear \
    --pool_size 10 --meta_sta_epoch 20 --update_iter 200 \
    --epochs 120  --batch-size 128 --warmup-epochs 0 \
    --lr 0.5  --opt-eps 0.001 \
    --color-jitter 0.06 --drop 0. -j 8 --num-classes 1000 --flops_minimum 0 --flops_maximum 600
