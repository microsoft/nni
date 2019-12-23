# Neural Architecture Search (NAS) on NNI

Automatic neural architecture search is taking an increasingly important role on finding better models. Recent research works have proved the feasibility of automatic NAS, and also found some models that could beat manually designed and tuned models. Some of representative works are [NASNet][2], [ENAS][1], [DARTS][3], [Network Morphism][4], and [Evolution][5]. There are new innovations keeping emerging.

However, it takes great efforts to implement NAS algorithms, and it is hard to reuse code base of existing algorithms in new one. To facilitate NAS innovations (e.g., design and implement new NAS models, compare different NAS models side-by-side), an easy-to-use and flexible programming interface is crucial.

With this motivation, our ambition is to provide a unified architecture in NNI, to accelerate innovations on NAS, and apply state-of-art algorithms on real world problems faster.

With [the unified interface](./NasInterface.md), there are two different modes for the architecture search. [The one](#supported-one-shot-nas-algorithms) is the so-called one-shot NAS, where a super-net is built based on search space, and using one shot training to generate good-performing child model. [The other](./NasInterface.md#classic-distributed-search) is the traditional searching approach, where each child model in search space runs as an independent trial, the performance result is sent to tuner and the tuner generates new child model.

* [Supported One-shot NAS Algorithms](#supported-one-shot-nas-algorithms)
* [Classic Distributed NAS with NNI experiment](./NasInterface.md#classic-distributed-search)
* [NNI NAS Programming Interface](./NasInterface.md)

## Supported One-shot NAS Algorithms

NNI supports below NAS algorithms now and being adding more. User can reproduce an algorithm or use it on owned dataset. we also encourage user to implement other algorithms with [NNI API](#use-nni-api), to benefit more people.

|Name|Brief Introduction of Algorithm|
|---|---|
| [ENAS](#enas) | Efficient Neural Architecture Search via Parameter Sharing [Reference Paper][1] |
| [DARTS](#darts) | DARTS: Differentiable Architecture Search [Reference Paper][3] |
| [P-DARTS](#p-darts) | Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation [Reference Paper](https://arxiv.org/abs/1904.12760)|

Note, these algorithms run **standalone without nnictl**, and supports PyTorch only. Tensorflow 2.0 will be supported in future release.

### Dependencies

* NNI 1.2+
* tensorboard
* PyTorch 1.2+
* git

### ENAS

[Efficient Neural Architecture Search via Parameter Sharing][1]. In ENAS, a controller learns to discover neural network architectures by searching for an optimal subgraph within a large computational graph. It uses parameter sharing between child models to achieve fast speed and excellent performance.

#### Usage

ENAS in NNI is still under development and we only support search phase for macro/micro search space on CIFAR10. Training from scratch and search space on PTB has not been finished yet. [Detailed Description](ENAS.md)

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/enas

# search in macro search space
python3 search.py --search-for macro

# search in micro search space
python3 search.py --search-for micro

# view more options for search
python3 search.py -h
```

### DARTS

The main contribution of [DARTS: Differentiable Architecture Search][3] on algorithm is to introduce a novel algorithm for differentiable network architecture search on bilevel optimization. [Detailed Description](DARTS.md)

#### Usage

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/darts
python3 search.py

# train the best architecture
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_49.json
```

### P-DARTS

[Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) bases on [DARTS](#DARTS). It's contribution on algorithm is to introduce an efficient algorithm which allows the depth of searched architectures to grow gradually during the training procedure.

#### Usage

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/pdarts
python3 search.py

# train the best architecture, it's the same progress as darts.
cd ../darts
python3 retrain.py --arc-checkpoint ../pdarts/checkpoints/epoch_2.json
```

## Use NNI API

NOTE, we are trying to support various NAS algorithms with unified programming interface, and it's in very experimental stage. It means the current programing interface may be updated in future.

### Programming interface

The programming interface of designing and searching a model is often demanded in two scenarios.

1. When designing a neural network, there may be multiple operation choices on a layer, sub-model, or connection, and it's undetermined which one or combination performs  best. So, it needs an easy way to express the candidate layers or sub-models.
2. When applying NAS on a neural network, it needs an unified way to express the search space of architectures, so that it doesn't need to update trial code for different searching algorithms.

NNI proposed API is [here](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch). And [here](https://github.com/microsoft/nni/tree/master/examples/nas/darts) is an example of NAS implementation, which bases on NNI proposed interface.

[1]: https://arxiv.org/abs/1802.03268
[2]: https://arxiv.org/abs/1707.07012
[3]: https://arxiv.org/abs/1806.09055
[4]: https://arxiv.org/abs/1806.10282
[5]: https://arxiv.org/abs/1703.01041
