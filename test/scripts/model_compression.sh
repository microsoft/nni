#!/bin/bash
set -e
CWD=${PWD}

echo ""
echo "===========================Testing: pruning and speedup==========================="
cd ${CWD}/../examples/model_compress

echo "testing slim pruning and speedup..."
python3 model_prune_torch.py --pruner_name slim --pretrain_epochs 1 --prune_epochs 1
python3 model_speedup.py --example_name slim --model_checkpoint ./checkpoints/pruned_vgg19_cifar10_slim.pth \
    --masks_file ./checkpoints/mask_vgg19_cifar10_slim.pth

echo "testing l1 pruning and speedup..."
python3 model_prune_torch.py --pruner_name l1 --pretrain_epochs 1 --prune_epochs 1
python3 model_speedup.py --example_name l1filter --model_checkpoint ./checkpoints/pruned_vgg16_cifar10_l1.pth \
    --masks_file ./checkpoints/mask_vgg16_cifar10_l1.pth

echo "testing apoz pruning and speedup..."
python3 model_prune_torch.py --pruner_name apoz --pretrain_epochs 1 --prune_epochs 1
python3 model_speedup.py --example_name apoz --model_checkpoint ./checkpoints/pruned_vgg16_cifar10_apoz.pth \
    --masks_file ./checkpoints/mask_vgg16_cifar10_apoz.pth

for name in level fpgm mean_activation
do
    echo "testing $name pruning..."
    python3 model_prune_torch.py --pruner_name $name --pretrain_epochs 1 --prune_epochs 1
done

#echo "testing lottery ticket pruning..."
#python3 lottery_torch_mnist_fc.py

echo ""
echo "===========================Testing: quantizers==========================="

echo "testing QAT quantizer..."
python3 QAT_torch_quantizer.py

echo "testing DoReFa quantizer..."
python3 DoReFaQuantizer_torch_mnist.py

echo "testing BNN quantizer..."
python3 BNN_quantizer_cifar10.py
