# Compressor
NNI provides easy-to-use toolkit to help user  design and use compression algorithm.

## Framework
We use the instrumentation method to insert a node or function after the corresponding position in the model.

When compression algorithm designer implements one prune algorithm, he only need to pay attention to the generation method of mask, without caring about applying the mask to the garph.
## Algorithm
We now provide some naive compression algorithm and four popular compress agorithms for users, including two pruning algorithm and two quantization algorithm.
Below is a list of model compression algorithms supported in our compressor
|Name|Paper|
|---|---|
| AGPruner| [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)|
| SensitivityPruner |[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)|
| QATquantizer      |[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)|
| DoReFaQuantizer   |[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)|

## Usage

Take naive level pruner as an example

If you want to prune all weight to 80% sparsity, you can add code below into your code before your training code.

Tensorflow code
```
nni.compressors.tfCompressor.LevelPruner(sparsity=0.8).compress(model_graph)
```

Pytorch code
```
nni.compressors.torchCompressor.LevelPruner(sparsity=0.8).compress(model)
```

Our compressor will automatically insert mask into your model, and you can train your model with masks without changing your training code. You will get a compressed model when you finish your training.

You can get more information in Algorithm details



