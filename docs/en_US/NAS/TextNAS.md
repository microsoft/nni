# TextNAS

## Introduction

This is the implementation of the TextNAS algorithm proposed in the paper [TextNAS: A Neural Architecture Search Space tailored for Text Representation](https://arxiv.org/pdf/1912.10729.pdf). TextNAS is a neural architecture search algorithm tailored for text representation, more specifically, TextNAS is based on a novel search space consists of operators widely adopted to solve various NLP tasks, and TextNAS also supports multi-path ensemble within a single network to balance the width and depth of the architecture. 

The search space of TextNAS contains: 

    * 1-D convolutional operator with filter size 1, 3, 5, 7 
    * recurrent operator (bi-directional GRU) 
    * self-attention operator
    * pooling operator (max/average)

Following the ENAS algorithm, TextNAS also utilizes parameter sharing to accelerate the search speed and adopts a reinforcement-learning controller for the architecture sampling and generation. Please refer to the paper for more details of TextNAS.

## Preparation

Prepare the word vectors and SST dataset, and organize them in data directory as shown below:

```
textnas
├── data
│   ├── sst
│   │   └── trees
│   │       ├── dev.txt
│   │       ├── test.txt
│   │       └── train.txt
│   └── glove.840B.300d.txt
├── dataloader.py
├── model.py
├── ops.py
├── README.md
├── search.py
└── utils.py
```

The following link might be helpful for finding and downloading the corresponding dataset:

* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
  * [glove.840B.300d.txt](http://nlp.stanford.edu/data/glove.840B.300d.zip)
* [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/sentiment/)
  * [trainDevTestTrees_PTB.zip](https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip)

## Examples

### Search Space

[Example code](https://github.com/microsoft/nni/tree/v1.9/examples/nas/textnas)

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/textnas

# view more options for search
python3 search.py -h
```

After each search epoch, 10 sampled architectures will be tested directly. Their performances are expected to be 40% - 42% after 10 epochs.

By default, 20 sampled architectures will be exported into `checkpoints` directory for next step.

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

TextNAS directly uses EnasTrainer, please refer to [ENAS](./ENAS.md) for the trainer APIs.
