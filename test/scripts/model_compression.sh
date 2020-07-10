#!/bin/bash
set -e
CWD=${PWD}

echo ""
echo "===========================Testing: pruning and speedup==========================="
cd ${CWD}/../examples/model_compress

for name in fpgm slim l1filter apoz
do
    echo "testing $name pruning and speedup..."
    python3 model_prune_torch.py --pruner_name $name --pretrain_epochs 1 --prune_epochs 1
    python3 model_speedup.py --example_name $name
done

for name in level mean_activation
do
    echo "testing $name pruning..."
    python3 model_prune_torch.py --pruner_name $name --pretrain_epochs 1 --prune_epochs 1
done

echo 'testing level pruner pruning'
python3 model_prune_torch.py --pruner_name level --pretrain_epochs 1 --prune_epochs 1

echo 'testing agp pruning'
python3 model_prune_torch.py --pruner_name agp --pretrain_epochs 1 --prune_epochs 2

echo 'testing mean_activation pruning'
python3 model_prune_torch.py --pruner_name mean_activation --pretrain_epochs 1 --prune_epochs 1

echo "testing lottery ticket pruning..."
python3 lottery_torch_mnist_fc.py --train_epochs 1

echo ""
echo "===========================Testing: quantizers==========================="
# to be enabled
#echo "testing QAT quantizer..."
#python3 QAT_torch_quantizer.py

#echo "testing DoReFa quantizer..."
#python3 DoReFaQuantizer_torch_mnist.py

#echo "testing BNN quantizer..."
#python3 BNN_quantizer_cifar10.py

rm -rf ./checkpoints/*
