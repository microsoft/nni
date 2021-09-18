#!/bin/bash
set -e
CWD=${PWD}

echo ""
echo "===========================Testing: NAS==========================="
EXAMPLE_DIR=${CWD}/../examples/nas

echo "testing mnist..."
cd $EXAMPLE_DIR/multi-trial/mnist
python3 search.py

echo "testing darts..."
cd $EXAMPLE_DIR/oneshot/darts
python3 search.py --epochs 1 --channels 2 --layers 4
python3 retrain.py --arc-checkpoint ./checkpoint.json --layers 4 --epochs 1

echo "testing enas..."
cd $EXAMPLE_DIR/oneshot/enas
python3 search.py --search-for macro --epochs 1
python3 search.py --search-for micro --epochs 1

#disabled for now
#echo "testing naive..."
#cd $EXAMPLE_DIR/naive
#python3 train.py

#echo "testing pdarts..."
#cd $EXAMPLE_DIR/legacy/pdarts
#python3 search.py --epochs 1 --channels 4 --nodes 2 --log-frequency 10 --add_layers 0 --add_layers 1 --dropped_ops 3 --dropped_ops 3
