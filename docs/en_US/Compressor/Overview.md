# Compressor
NNI provides an easy-to-use toolkit to help user design and use compression algorithms. It supports Tensorflow and PyTorch with unified interface. For users to compress their models, they only need to add several lines in their code. There are some popular model compression algorithms built-in in NNI. Users could further use NNI's auto tuning power to find the best compressed model, which is detailed in [Auto Model Compression](./AutoCompression). On the other hand, users could easily customize their new compression algorithms using NNI's interface, refer to the tutorial [here](#CustomizeCompression).

## Supported algorithms
We have provided two naive compression algorithms and four popular ones for users, including three pruning algorithms and three quantization algorithms:

|Name|Brief Introduction of Algorithm|
|---|---|
| [LevelPruner](./Pruner#LevelPruner) | Pruning the specified ratio on each weight based on absolute values of weights |
| [AGPruner](./Pruner#AGPruner) | To prune, or not to prune: exploring the efficacy of pruning for model compression. [Reference Paper](https://arxiv.org/abs/1710.01878)|
| [SensitivityPruner](./Pruner#SensitivityPruner) | Learning both Weights and Connections for Efficient Neural Networks. [Reference Paper](https://arxiv.org/abs/1506.02626)|
| [NaiveQuantizer](./Quantizer#NaiveQuantizer) |  Quantize weights to default 8 bits |
| [QATquantizer](./Quantizer#QATquantizer) | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. [Reference Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)|
| [DoReFaQuantizer](./Quantizer#DoReFaQuantizer) | DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. [Reference Paper](https://arxiv.org/abs/1606.06160)|

## Usage of built-in compression algorithms

We use a simple example to show how to modify your trial code in order to apply the compression algorithms. Let's say you want to prune all weight to 80% sparsity with LevelPruner, you can add the following three lines into your code before training your model ([here](https://github.com/microsoft/nni/tree/master/examples/model_compress) is complete code).

Tensorflow code
```
config = [{'sparsity':0.8,'support_type': 'default'}]
pruner = nni.compressors.tf_compressor.LevelPruner(config)
pruner(tf.get_default_graph())
```

Pytorch code
```
config = [{'sparsity':0.8,'support_type': 'default'}]
pruner = nni.compressors.torch_compressor.LevelPruner(config)
pruner(model)
```

You can use other compression algorithms in the package of `nni.compressors`. The algorithms are implemented in both Pytorch and Tensorflow, under `nni.compressors.torch_compressor` and `nni.compressors.tf_compressor` respectively. You can refer to [Pruner](./Pruner) and [Quantizer](./Quantizer) for detail description of supported algorithms.

The function call `pruner(model)` receives user defined model (in Tensorflow the model can be obtained with `tf.get_default_graph`, while in Pytorch the model is the defined model class), and the model is modified with masks inserted. Then when you run the model, the masks take effect. The masks can be adjusted at runtime by the algorithms.

When instantiate a compression algorithm, there is `config` passed in. We describe how to write this config below.

### User configuration for a compression algorithm

When compressing a model, users may want to specify the ratio for sparsity, to specify different ratios for different types of operations, to exclude certain types of operations, or to compress only a certain types of operations. For users to express these kinds of requirements, we define a configuration specification. It can be seen as a python `list` object, where each element is a `dict` object. In each `dict`, there are some keys commonly supported by NNI compression:

* __op_types__: This is to specify what types of operations to be compressed. 'default' means following the algorithm's default setting.
* __op_names__: This is to specify by name what operations to be compressed. If this field is omitted, operations will not be filtered by it.
* __exclude__: Default is False. If this field is True, it means the operations with specified types and names will be excluded from the compression.

There are also other keys in the `dict`, but they are specific for every compression algorithm. For example, some , some.

The `dict`s in the `list` are applied one by one, that is, the configurations in latter `dict` will overwrite the configurations in former ones on the operations that are within the scope of both of them. 

A simple example of configuration is shown below:
```
[{
    'sparsity': 0.8,
    'op_types': 'default'
},
{
    'sparsity': 0.6,
    'op_names': ['op_name1', 'op_name2']
},
{
    'exclude': True,
    'op_names': ['op_name3']
}]
```
It means following the algorithm's default setting for compressed operations with sparsity 0.8, but for `op_name1` and `op_name2` use sparsity 0.6, and please do not compress `op_name3`.

### Other APIs

Some compression algorithms use epochs to control the progress of compression, and some algorithms need to do something after every minibatch. Therefore, we provide another two APIs for users to invoke. One is `update_epoch`, you can use it as follows:

Tensorflow code 
```
pruner.update_epoch(epoch, sess)
```
Pytorch code
```
pruner.update_epoch(epoch)
```

The other is `step`, it can be called with `pruner.step()` after each minibatch. Note that not all algorithms need these two APIs, for those that do not need them, calling them is allowed but has no effect.

__[TODO]__ The last API is for users to export the compressed model. You will get a compressed model when you finish the training using this API. It also exports another file storing the values of masks.

<a name="CustomizeCompression"></a>

## Customize new compression algorithms

To simplify writing a new compression algorithm, we design programming interfaces which are simple but flexible enough. There are interfaces for pruner and quantizer respectively.

### Pruning algorithm

If you want to write a new pruning algorithm, you can write a class that inherits `nni.compressors.tf_compressor.TfPruner` or `nni.compressors.torch_compressor.TorchPruner` depending on which framework you use. Then, override the member functions with the logic of your algorithm.

```
# This is writing a pruner in tensorflow.
# For writing a pruner in Pytorch, you can simply replace
# nni.compressors.tf_compressor.TfPruner with
# nni.compressors.torch_compressor.TorchPruner
class YourPruner(nni.compressors.tf_compressor.TfPruner):
    def __init__(self, config):
        # suggest you to use the NNI defined spec for config
        pass
    
    def bind_model(self, model):
        # this func can be used to remember the model or its weights
        # in member variables, for getting their values during training
        pass

    def calc_mask(self, layer_info, weight):
        # you can get layer name in layer_info.name
        # you can get weight data in weight
        # design your mask and return your mask
        return your_mask

    # note for pytorch version, there is no sess in input arguments
    def update_epoch(self, epoch_num, sess):
        pass

    # note for pytorch version, there is no sess in input arguments
    def step(self, sess):
        # can do some processing based on the model or weights binded
        # in the func bind_model
        pass
```

For the simpliest algorithm, you only need to override `calc_mask`, it receives each layer's information (i.e., layer name) and the layer's weight, you generate the mask for this weight in this function and return. Then NNI applies the mask for you.

Some algorithms generate mask based on training progress, i.e., epoch number. We provide `update_epoch` for the pruner to be aware of the training progress.

Some algorithms may want global information for generating masks, for example, all weights of the model (for statistic information), model optimizer's information. NNI supports this requirement using `bind_model`. `bind_model` receives the complete model, thus, it could record any information (e.g., reference to weights) it cares about. Then `step` can process or update the information according to the algorithm. You can refer to [source code of built-in algorithms](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/compressors) for example implementations.

### Quantization algorithm

The interface for customizing quantization algorithm is similar to that of pruning algorithms. The only difference is that `calc_mask` is replaced with `quantize_weight`. `quantize_weight` directly returns the quantized weights rather than mask, because for quantization the quantized weights cannot be obtained by applying mask.

```
# This is writing a Quantizer in tensorflow.
# For writing a Quantizer in Pytorch, you can simply replace
# nni.compressors.tf_compressor.TfQuantizer with
# nni.compressors.torch_compressor.TorchQuantizer
class YourPruner(nni.compressors.tf_compressor.TfQuantizer):
    def __init__(self, config):
        # suggest you to use the NNI defined spec for config
        pass

    def bind_model(self, model):
        # this func can be used to remember the model or its weights
        # in member variables, for getting their values during training
        pass

    def quantize_weight(self, layer_info, weight):
        # you can get layer name in layer_info.name
        # you can get weight data in weight
        # design your quantizer and return new weight
        return new_weight

    # note for pytorch version, there is no sess in input arguments
    def update_epoch(self, epoch_num, sess):
        pass

    # note for pytorch version, there is no sess in input arguments
    def step(self, sess):
        # can do some processing based on the model or weights binded
        # in the func bind_model
        pass
    
    # you can also design your method
    def your_method(self, your_input):
        #your code
    
    def bind_model(self, model):
        #preprocess model
```

__[TODO]__ Will add another member function `quantize_layer_output`, as some quantization algorithms also quantize layers' output.

### Usage of user customized compression algorithm

__[TODO]__ ...
