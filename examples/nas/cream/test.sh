CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./examples/nas/cream/distributed_test.sh 8 \
    --data ./data/imagenet --model_selection 285 --resume ./data/ckpts/285.pth.tar
