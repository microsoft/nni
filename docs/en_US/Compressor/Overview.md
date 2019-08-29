# Compressor
NNI provides an easy-to-use toolkit to help user design and use compression algorithms. It supports Tensorflow and Pytorch with unified interface. For users to compress their models, they only need to add several lines in their code. There are some popular model compression algorithms built-in in NNI. Users could further use NNI's auto tuning power to find the best compressed model, which is detailed [here](./AutoCompression). On the other hand, users could easily customize their new compression algorithms using NNI's interface, refer to the tutorial [here](#CustomizeCompression).

## Supported algorithms
We have provided two naive compression algorithms and four popular ones for users, including three pruning algorithms and three quantization algorithms:

|Name|Brief Introduction of Algorithm|
|---|---|
| [LevelPruner](./Pruner#LevelPruner) | None |
| [AGPruner](./Pruner#AGPruner) | To prune, or not to prune: exploring the efficacy of pruning for model compression. [Reference Paper](https://arxiv.org/abs/1710.01878)|
| [SensitivityPruner](./Pruner#SensitivityPruner) | Learning both Weights and Connections for Efficient Neural Networks. [Reference Paper](https://arxiv.org/abs/1506.02626)|
| [NaiveQuantizer](./Quantizer#NaiveQuantizer) | None |
| [QATquantizer](./Quantizer#QATquantizer) | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. [Reference Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)|
| [DoReFaQuantizer](./Quantizer#DoReFaQuantizer) | DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. [Reference Paper](https://arxiv.org/abs/1606.06160)|

## Usage of built-in compression algorithms

We use a simple example to show how to modify your trial code in order to apply the compression algorithms. Let's say you want to prune all weight to 80% sparsity with LevelPruner, you can add the following three lines into your code before training your model ([here]() is complete code).

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

There are also other keys in the `dict`, but they are specific for every compression algorithms. For example, some , some.

The configuration in each `dict` is applied one by one, that is, latter configuration will overwrite former ones on the operations that are within the scope of both of them.

Code 
```
[{
    'sparsity':0.8,
    'support_layer':'default'
    'support_op':['op_name1','op_name2']
}]
```

file
```
AGPruner:       
  config:
    -
        start_epoch: 0
        end_epoch: 16
        frequency: 2
        initial_sparsity: 0.05
        final_sparsity: 0.60
        support_type: default
    - 
        prune: False
        start_epoch: 0
        end_epoch: 20
        frequency: 2
        initial_sparsity: 0.05
        final_sparsity: 0.60
        support_type: [Linear] 
        support_op: [conv1, conv2]
```

For Take naive level pruner as an example, you can get detailed information in Pruner details.
Our compressor will automatically insert mask into your model, and you can train your model with masks without changing your training code. You will get a compressed model when you finish your training.

You can get more information in Algorithm details

<a name="CustomizeCompression"></a>

## Customize new compression algorithms

We use the instrumentation method to insert a node or function after the corresponding position in the model.  And we provide interface for designer to design compression algorithm easily.

If you want to use mask to prune a model, you can use Pruner as base class. And design your mask in calc_mask() method.

Tensorflow code
```
class YourPruner(nni.compressors.tf_compressor.TfPruner):
    def __init__(self, your_input):
        # defaultly, we suggest you to use configure_list as input
        pass
    
    # don not change calc_mask() input 
    def calc_mask(self, layer_info, weight):
        # you can get layer name in layer_info.name
        # you can get weight data in weight
        # design your mask and return your mask
        return your_mask
    
    # you can also design your method help to generate mask
    def your_method(self, your_input):
        #your code

```
Pytorch code
```
class YourPruner(nni.compressors.torch_compressor.TorchPruner):
    def __init__(self, your_input):
        pass
    
    # don not change calc_mask() input 
    def calc_mask(self, layer_info, weight):
        # you can get layer name in layer_info.name
        # you can get weight data in weight
        # design your mask and return your mask
        return your_mask
    
    # you can also design your method help to generate mask
    def your_method(self, your_input):
        #your code
```

if you want to generate new weight and replace the old one, you are suggested to use Quantizer as base class. And you can manage the weight in quantize_weight() method.

Tensorflow code
```
class YourPruner(nni.compressors.tf_compressor.TfQuantizer):
    def __init__(self, your_input):
        pass
    
    # don not change quantize_weight() input 
    def quantize_weight(self, layer_info, weight):
        # you can get layer name in layer_info.name
        # you can get weight data in weight
        # design your quantizer and return new weight
        return new_weight
    
    # you can also design your method
    def your_method(self, your_input):
        #your code

```
Pytorch code
```
class YourPruner(nni.compressors.torch_compressor.TorchQuantizer):
    def __init__(self, your_input):
        pass
    
    # don not change quantize_weight() input 
    def quantize_weight(self, layer_info, weight):
        # you can get layer name in layer_info.name
        # you can get weight data in weight
        # design your quantizer and return new weight
        return new_weight
    
    # you can also design your method
    def your_method(self, your_input):
        #your code
```

#### Preprocess Model
Sometimes, designer wants to preprocess model before compress, designer can overload preprocess_model() method 

```
class YourPruner(nni.compressors.torch_compressor.TorchQuantizer):
    def __init__(self, your_input):
        pass
    
    # don not change quantize_weight() input 
    def quantize_weight(self, layer_info, weight):
        # you can get layer name in layer_info.name
        # you can get weight data in weight
        # design your quantizer and return new weight
        return new_weight
    
    # you can also design your method
    def your_method(self, your_input):
        #your code
    
    def preprocess_model(self, model):
        #preprocess model
```
#### Step and Epoch
if an designer wants to update mask every step,  designer can implement step() or update_epoch() method in his code, and tell user to call when use it.

