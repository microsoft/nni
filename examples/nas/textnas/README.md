# TextNAS: A Neural Architecture Search Space tailored for Text Representation

TextNAS by MSRA. Official Release.

[Paper link](https://arxiv.org/abs/1912.10729)

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
├── final_arc.json
├── model.py
├── ops.py
├── README.md
├── retrain.py
├── search.py
└── utils.py
```

The following link might be helpful for finding and downloading the corresponding dataset:

* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
* [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/sentiment/)

## Search

```
python search.py
```

After each search epoch, 10 sampled architectures will be tested directly. Their performances are expected to be 40% - 42% after 10 epochs.

By default, 20 sampled architectures will be exported into `checkpoints` directory for next step.

## Retrain

Note that we have already provided one of our searched architecture as `final_arc.json`, which you can directly use if you have skipped the last step.

Retrain any architecture you want with

```
python retrain.py --arc-checkpoint /path/to/your/arc
```

The default hyper-parameters are expected to give you an accuracy between 51% and 52%. 52.67% can be achieved when trained in detail.
