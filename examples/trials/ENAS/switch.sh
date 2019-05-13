#!/bin/bash
# Use this to swith from the subgraph version and whole graph version of ENAS

switch_to_subgraph()
{
    sed -i 's/platform\:\ tensorflow/platform\:\ others/g' trial_code/src/cifar10/general_child.py
    sed -i 's/wholegraph/subgraph/g' trial_code/macro_cifar10.sh
}
switch_to_whole_graph()
{
    sed -i 's/platform\:\ others/platform\:\ tensorflow/g' trial_code/src/cifar10/general_child.py
    sed -i 's/subgraph/wholegraph/g' trial_code/macro_cifar10.sh
}

echo "Swith to : [subgraph or wholegraph]"
read CONFIRM
if [ "$CONFIRM" = "subgraph" ]
then
    switch_to_subgraph
elif [ "$CONFIRM" = "wholegraph" ]
then
    switch_to_whole_graph
else
    echo "Invalid Selection!"
fi
