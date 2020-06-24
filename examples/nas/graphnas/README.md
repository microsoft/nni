# GraphNAS: Graph Neural Architecture Search with Reinforcement Learning
Graph Neural Networks (GNNs) are an important part of neural networks, which have achieved great success in graph-structured data, such as social networks and biological data. There has been some studies about how to apply Neural Architecture Search (NAS) to GNNs, such as GraphNAS and Auto-GNN.

This example implements the [GraphNAS algorithm](https://arxiv.org/abs/1904.09981) with a [GraphNASTuner](). This implementation on NNI is based on the [official implementation](https://github.com/GraphNAS/GraphNAS). For now, this example supports classic NAS with data in [torch-geometric](https://github.com/rusty1s/pytorch_geometric) format.

# Requirements
To run this example, you need to have a python environment with these packages installed:
```
torch # https://pytorch.org/
torch-scatter
torch-sparse
torch-cluster
torch-geometric # https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
nni
```
You can follow the corresponding installation instruction of torch* to install them properly. For nni, just install it with ```pip```.

# Run this example
Clone this repo:
```
git clone https://github.com/microsoft/nni.git
cd nni/examples/nas/graphnas
```
Controller training:
```
nnictl create --config config.yml
```
Retraining:
```
python3 retrain.py --dataset Citeseer
```
You can modify ```config.yml``` to specific the dataset (one of {```Cora```, ```Citeseer```, ```Pubmed```}) as well as the path of saved architectures. Change ```gpuNum`` to a positive integer to accelerate if you have a GPU.

