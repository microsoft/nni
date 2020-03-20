### Config for pruners excluding SlimPruner
# python main.py --config configs/pruner/densenet40_pruner.yaml --save_name Task_01 --random_seed 2222 --pruner_name GradientWeightRankFilterPruner --pretrain ./experiments/pretrained_models/densenet40_pretrain_cifar10_seed_2333.pth

### Config for SlimPruner
# python main.py --config configs/pruner/densenet40_slim_pruner.yaml --save_name Task_01--random_seed 2333 --pruner_name SlimPruner --pretrain ./experiments/pretrained_models/densenet40_pretrain_cifar10_seed_2333.pth
