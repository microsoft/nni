#!/bin/bash

# download automlbenchmark repository
if [ ! -d './automlbenchmark' ] ; then
    git clone https://github.com/openml/automlbenchmark.git --branch v1.6 --depth 1
fi

# install dependencies 
pip3 install -r automlbenchmark/requirements.txt
pip3 install -r requirements.txt --ignore-installed
