#!/bin/bash
set -e
CWD=${PWD}

echo ""
echo "===========================Testing: NAS==========================="
EXAMPLE_DIR=${CWD}/../examples/nas

echo "testing nnictl ss_gen (classic nas)..."
cd $EXAMPLE_DIR/classic_nas
SEARCH_SPACE_JSON=nni_auto_gen_search_space.json
if [ -f $SEARCH_SPACE_JSON ]; then
    rm $SEARCH_SPACE_JSON
fi
nnictl ss_gen -t "python3 mnist.py"
if [ ! -f $SEARCH_SPACE_JSON ]; then
    echo "Search space file not found!"
    exit 1
fi

echo "testing darts..."
cd $EXAMPLE_DIR/darts
python3 search.py --epochs 1 --channels 2 --layers 4
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_0.json --layers 4 --epochs 1

echo "testing enas..."
cd $EXAMPLE_DIR/enas
python3 search.py --search-for macro --epochs 1
python3 search.py --search-for micro --epochs 1

#disabled for now
#echo "testing naive..."
#cd $EXAMPLE_DIR/naive
#python3 train.py

echo "testing pdarts..."
cd $EXAMPLE_DIR/pdarts
python3 search.py --epochs 1 --channels 4 --nodes 2 --log-frequency 10 --add_layers 0 --add_layers 1 --dropped_ops 3 --dropped_ops 3
