# TextNAS

## Introduction

This is the implementation of the TextNAS algorithm proposed in the paper [TextNAS: A Neural Architecture Search Space tailored for Text Representation](https://arxiv.org/pdf/1912.10729.pdf). TextNAS is a neural architecture search algorithm tailored for text representation, more specifically, TextNAS is based on a novel search space consists of operators widely adopted to solve various NLP tasks, and TextNAS also supports multi-path ensemble within a single network to balance the width and depth of the architecture. 

The search space of TextNAS contains: 

    * 1-D convolutional operator with filter size 1, 3, 5, 7 
    * recurrent operator (bi-directional GRU) 
    * self-attention operator
    * pooling operator (max/average)

Following the ENAS algorithm, TextNAS also utilizes parameter sharing to accelerate the search speed and adopts a reinforcement-learning controller for the architecture sampling and generation. Please refer to the paper for more details of TextNAS.

## Examples

### Search Space

[Example code](https://github.com/microsoft/nni/tree/master/examples/nas/textnas)

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/textnas

# view more options for search
python3 search.py -h
```

### retrain

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/textnas

# default to retrain on sst-2
sh run_retrain.sh
```

## Reference

### PyTorch

```eval_rst
..  autoclass:: nni.nas.pytorch.enas.EnasTrainer
    :members:

..  autoclass:: nni.nas.pytorch.enas.EnasMutator
    :members:
```
