#!/bin/bash
set -e

python3 -m pip install scikit-learn==0.20.0 --user
python3 -m pip install torch==1.2.0 --user
python3 -m pip install torchvision==0.4.0 --user
python3 -m pip install keras==2.1.6 --user
python3 -m pip install tensorflow-gpu==1.12.0 --user
sudo apt-get install swig -y
PATH=$HOME/.local/bin:$PATH nnictl package install --name=SMAC
PATH=$HOME/.local/bin:$PATH nnictl package install --name=BOHB
