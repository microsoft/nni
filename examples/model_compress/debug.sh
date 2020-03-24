python main.py --config configs/pruner/vgg16_pruner.yaml --save_name debug --random_seed 2333 --pruner_name ActivationAPoZRankFilterPruner --pretrain ./experiments/pretrained_models/vgg16_pretrain_cifar10_seed_2333.pth

# python main.py --config configs/pruner/resnet56_slim_pruner.yaml --save_name debug --random_seed 2333 --pruner_name SlimPruner --pretrain ./experiments/pretrained_models/resnet56_pretrain_cifar10_seed_2333.pth

# python model_speedup.py --example_name l1filter --masks_file /home/yanchenqian/workspace/nni-dev/examples/model_compress/experiments/vgg16_cifar10_L1FilterPruner/Task_01_seed_2333/checkpoints/mask.pth --model_checkpoint /home/yanchenqian/workspace/nni-dev/examples/model_compress/experiments/vgg16_cifar10_L1FilterPruner/Task_01_seed_2333/checkpoints/pruned.pth

# python main.py --config configs/pruner/naive.yaml  --save_name debug --random_seed 2222 --pruner_name ActivationAPoZRankFilterPruner --pretrain ./experiments/pretrained_models/resnet56_pretrain_cifar10_seed_2333.pth

# cp ./experiments/resnet56_cifar10_FPGMPruner/Task_01_seed_2222/checkpoints/pretrain.pth ./experiments/pretrained_models/resnet56_pretrain_cifar10_seed_2222.pth

# declare -a MODELS=('resnet56' 'vgg16' 'densenet40')
# declare -a SEEDS=('2333' '2222' '1000')
# for model in "${MODELS[@]}"
# do
#   for seed in "${SEEDS[@]}" 
#   do 
#     cp "./experiments/"$model"_cifar10_FPGMPruner/Task_01_seed_$seed/checkpoints/pretrain.pth" ."/experiments/pretrained_models/$model"_pretrain_cifar10_seed_"$seed.pth"
#   done
# done

# python main.py --config configs/pruner/naive.yaml  --save_name debug --random_seed 2222 --pruner_name ActivationAPoZRankFilterPruner --pretrain ./examples/model_compress/experiments/naive_mnist_GradientWeightRankFilterPruner/debug_seed_2222/checkpoints/pretrain.pth