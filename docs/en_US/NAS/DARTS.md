# DARTS

## Introduction

The paper [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) addresses the scalability challenge of architecture search by formulating the task in a differentiable manner. Their method is based on the continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent.

Authors' code optimizes the network weights and architecture weights alternatively in mini-batches. They further explore the possibility that uses second order optimization (unroll) instead of first order, to improve the performance.

Implementation on NNI is based on the [official implementation](https://github.com/quark0/darts) and a [popular 3rd-party repo](https://github.com/khanrc/pt.darts). DARTS on NNI is designed to be general for arbitrary search space. A CNN search space tailored for CIFAR10, same as the original paper, is implemented as a use case of DARTS.

## Reproduction Results

The above-mentioned example is meant to reproduce the results in the paper, we do experiments with first and second order optimization. Due to the time limit, we retrain *only the best architecture* derived from the search phase and we repeat the experiment *only once*. Our results is currently on par with the results reported in paper. We will add more results later when ready.

|                        | In paper      | Reproduction |
| ---------------------- | ------------- | ------------ |
| First order (CIFAR10)  | 3.00 +/- 0.14 | 2.78         |
| Second order (CIFAR10) | 2.76 +/- 0.09 | 2.80         |

## Examples

### CNN Search Space

[Example code](https://github.com/microsoft/nni/tree/v1.9/examples/nas/darts)

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/darts
python3 search.py

# train the best architecture
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_49.json
```

## Reference

### PyTorch

```eval_rst
..  autoclass:: nni.algorithms.nas.pytorch.darts.DartsTrainer
    :members:

..  autoclass:: nni.algorithms.nas.pytorch.darts.DartsMutator
    :members:
```

## Limitations

* DARTS doesn't support DataParallel and needs to be customized in order to support DistributedDataParallel.
