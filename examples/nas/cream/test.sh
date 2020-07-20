CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./examples/nas/cream/distributed_test.sh 8 \
--data ~/data_local/imagenet --model_selection 285 --resume ~/data_local/nips_ckp/285m/model_best.pth.tar # 0.06 --drop 0.  -j 8 --num-classes 1000 --flops_minimum 0 --flops_maximum 600
