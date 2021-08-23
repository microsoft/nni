#!/bin/bash
set -e
CWD=${PWD}

echo ""
echo "===========================Testing: pruning and speedup==========================="
cd ${CWD}/../examples/model_compress

echo "testing fpgm pruning and speedup..."
python3 pruning/basic_pruners_torch.py --pruner fpgm --pretrain-epochs 1 --fine-tune-epochs 1 --model vgg16 --dataset cifar10
python3 pruning/model_speedup.py --example_name fpgm

echo "testing slim pruning and speedup..."
python3 pruning/basic_pruners_torch.py --pruner slim --pretrain-epochs 1 --fine-tune-epochs 1 --model vgg19 --dataset cifar10 --sparsity 0.7
python3 pruning/model_speedup.py --example_name slim

echo "testing l1filter pruning and speedup..."
python3 pruning/basic_pruners_torch.py --pruner l1filter --pretrain-epochs 1 --fine-tune-epochs 1 --model vgg16 --dataset cifar10
python3 pruning/model_speedup.py --example_name l1filter

echo "testing apoz pruning and speedup..."
python3 pruning/basic_pruners_torch.py --pruner apoz --pretrain-epochs 1 --fine-tune-epochs 1 --model vgg16 --dataset cifar10
python3 pruning/model_speedup.py --example_name apoz

echo 'testing level pruner pruning'
python3 pruning/basic_pruners_torch.py --pruner level --pretrain-epochs 1 --fine-tune-epochs 1 --model lenet --dataset mnist

echo 'testing agp pruning'
python3 pruning/basic_pruners_torch.py --pruner agp --pretrain-epochs 1 --fine-tune-epochs 1 --model lenet --dataset mnist

echo 'testing mean_activation pruning'
python3 pruning/basic_pruners_torch.py --pruner mean_activation --pretrain-epochs 1 --fine-tune-epochs 1 --model vgg16 --dataset cifar10

echo "testing lottery ticket pruning..."
python3 pruning/lottery_torch_mnist_fc.py --train_epochs 1

echo ""
echo "===========================Testing: quantizers==========================="
# to be enabled
#echo "testing QAT quantizer..."
#python3 QAT_torch_quantizer.py

#echo "testing DoReFa quantizer..."
#python3 DoReFaQuantizer_torch_mnist.py

#echo "testing BNN quantizer..."
#python3 BNN_quantizer_cifar10.py

rm -rf ./experiment_data/*
