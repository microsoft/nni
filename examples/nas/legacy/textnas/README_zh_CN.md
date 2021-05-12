# TextNAS: A Neural Architecture Search Space tailored for Text Representation

TextNAS 由 MSRA 提出 正式版本。

[论文链接](https://arxiv.org/abs/1912.10729)

## 准备

准备词向量和 SST 数据集，并按如下结构放到 data 目录中：

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

以下链接有助于查找和下载相应的数据集：

* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
* [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/sentiment/)

## 搜索

```
python search.py
```

在每个搜索 Epoch 后，会直接测试 10 个采样的结构。 10 个 Epoch 后的性能预计为 40% - 42%。

默认情况下，20 个采样结构会被导出到 `checkpoints` 目录中，以便进行下一步处理。

## 重新训练

```
sh run_retrain.sh
```

默认情况下，脚本会重新训练 SST-2 数据集上作者所提供的网络结构。
