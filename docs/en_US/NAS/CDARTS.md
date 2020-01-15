# CDARTS

## Introduction
CDARTS builds a cyclic feedback mechanism between the search and evaluation networks. First, the search network generates an initial topology for evaluation, so that the weights of the evaluation network can be optimized. Second, the architecture topology in the search network is further optimized by the label supervision in classification, as well as the regularization from the evaluation network through feature distillation. Repeating the above cycle results in a joint optimization of the search and evaluation networks, and thus enables the evolution of the topology to fit the final evaluation network.

## Reproduction Results
This is CDARTS based on the NNI platform, which currently supports CIFAR10 search and retrain. ImageNet search and retrain should also be supported, and we provide corresponding interfaces. Our reproduced results on NNI are slightly lower than the paper, but much higher than the original DARTS. Here we show the results of three independent experiments on CIFAR10.
| Runs | Paper | NNI | 
| ---- |:-------------:| :-----:|
| 1 | 97.52 | 97.44 |
| 2 | 97.53 | 97.48 |
| 3 | 97.58 | 97.56 |


## Examples

[Example code](https://github.com/microsoft/nni/tree/master/examples/nas/cdarts)

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# install apex for distributed training.
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext

# search the best architecture
cd examples/nas/cdarts
bash run_search_cifar.sh

# train the best architecture.
bash run_retrain_cifar.sh
```
