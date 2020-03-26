### Config for pruners excluding SlimPruner
python main.py --config configs/pruner/vgg16_pruner.yaml --save_name output --random_seed 2333 --pruner_name TaylorFOWeightFilterPruner --pretrain ./experiments/pretrained_models/vgg16_pretrain_cifar10_seed_2333.pth

### Config for SlimPruner
# python main.py --config configs/pruner/vgg16_slim_pruner.yaml --save_name output --random_seed 2333 --pruner_name SlimPruner --pretrain ./experiments/pretrained_models/vgg16_pretrain_cifar10_seed_2333.pth
