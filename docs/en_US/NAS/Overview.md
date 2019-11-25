# Neural Architecture Search (NAS) on NNI

Automatic neural architecture search is taking an increasingly important role on finding better models. Recent research works have proved the feasibility of automatic NAS, and also found some models that could beat manually designed and tuned models. Some of representative works are [NASNet][2], [ENAS][1], [DARTS][3], [Network Morphism][4], and [Evolution][5]. There are new innovations keeping emerging.

However, it takes great efforts to implement NAS algorithms, and it is hard to reuse code base of existing algorithms in new one. To facilitate NAS innovations (e.g., design and implement new NAS models, compare different NAS models side-by-side), an easy-to-use and flexible programming interface is crucial.

With this motivation, our ambition is to provide a unified architecture in NNI, to accelerate innovations on NAS, and apply state-of-art algorithms on real world problems faster.

* [Supported One-shot NAS Algorithms](#supported-one-shot-nas-algorithms)
* [NNI NAS Programming Interface](.NasInterface.md)
* [Classic Distributed NAS with NNI experiment](.ClassicNas.md)

## Supported One-shot NAS Algorithms

NNI supports below NAS algorithms now and being adding more. User can reproduce an algorithm or use it on owned dataset. we also encourage user to implement other algorithms with [NNI API](#use-nni-api), to benefit more people.

|Name|Brief Introduction of Algorithm|
|---|---|
| [ENAS](#enas) | DARTS: Differentiable Architecture Search [Reference Paper][3] |
| [P-DARTS](#p-darts) | Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation [Reference Paper](https://arxiv.org/abs/1904.12760)|
| [DARTS](#darts) | DARTS: Differentiable Architecture Search [Reference Paper][3] |
| [P-DARTS](#p-darts) | Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation [Reference Paper](https://arxiv.org/abs/1904.12760)|

Note, these algorithms run **standalone without nnictl**, and supports PyTorch only.

### Dependencies

* NNI 1.2+
* PyTorch 1.2+
* git

### DARTS

The main contribution of [DARTS: Differentiable Architecture Search][3] on algorithm is to introduce a novel algorithm for differentiable network architecture search on bilevel optimization.

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
cd examples/nas/darts
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_2.json
```

## Use NNI API

NOTE, we are trying to support various NAS algorithms with unified programming interface, and it's in very experimental stage. It means the current programing interface may be updated in future.

*previous [NAS annotation](../AdvancedFeature/GeneralNasInterfaces.md) interface will be deprecated soon.*

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
