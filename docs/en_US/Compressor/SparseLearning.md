Sparse Learning
===

In [Sparse Networks from Scratch: Faster Training without Losing Performance](https://arxiv.org/abs/1907.04840), author Tim Dettmers and Luke Zettlemoyer provide a algorithm to prune deep neural networks using smoothed gradients to identify layers and weights to be pruned.

You can find their orignal source code in their [github repo](https://github.com/TimDettmers/sparse_learning), we also follow their usage.

### Usage

####pytorch

You can run `python main.py --data DATASET_NAME --model MODEL_NAME` to run a model on MNIST (`--data mnist`) or CIFAR-10 (`--data cifar`).

The following models can be specified with the `--model` command out-of-the-box:
```
 MNIST:

	lenet5
	lenet300-100

 CIFAR-10:

	alexnet-s
	alexnet-b
	vgg-c
	vgg-d
	vgg-like
	wrn-28-2
	wrn-22-8
	wrn-16-8
	wrn-16-10
```

#### tensorflow

We also provide navie sparse learning pruner named NaiveSparsePruner.

```
from nni.compressors.tf_Compressor.sparse_pruner import NaiveSparsePruner, LinearDecay
pruner = NaiveSparsePruner(model.optimizer, LinearDecay(0.5,2000))
pruner.compress_default_graph()
```
LinearDecay means pruning rate linearly with each step.
LinearDecay(0.5, 2000) means its pruning rate decay from 0.5 to 0 in 2000 steps.
Defaultly, we set pruning density 0.05, you can use ```pruner = NaiveSparsePruner(model.optimizer, LinearDecay(0.5,2000), density)``` to change it.
