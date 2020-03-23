### Config for pruners excluding SlimPruner
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/pruner/densenet40_pruner.yaml --save_name Task_01 --random_seed 2333 --pruner_name TaylorFOWeightFilterGlobalPruner --pretrain ./experiments/pretrained_models/densenet40_pretrain_cifar10_seed_2333.pth

CUDA_VISIBLE_DEVICES=2 python main.py --config configs/pruner/densenet40_pruner.yaml --save_name Task_01 --random_seed 2222 --pruner_name TaylorFOWeightFilterGlobalPruner --pretrain ./experiments/pretrained_models/densenet40_pretrain_cifar10_seed_2222.pth

CUDA_VISIBLE_DEVICES=2 python main.py --config configs/pruner/densenet40_pruner.yaml --save_name Task_01 --random_seed 1000 --pruner_name TaylorFOWeightFilterGlobalPruner --pretrain ./experiments/pretrained_models/densenet40_pretrain_cifar10_seed_1000.pth

### Config for SlimPruner
# python main.py --config configs/pruner/densenet40_slim_pruner.yaml --save_name Task_01--random_seed 2333 --pruner_name SlimPruner --pretrain ./experiments/pretrained_models/densenet40_pretrain_cifar10_seed_2333.pth
