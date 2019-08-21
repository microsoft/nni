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
### For compression algorithm user

Take naive level pruner as an example

If you want to prune all weight to 80% sparsity, you can add code below into your code before your training code.

Tensorflow code
```
pruner = nni.compressors.tfCompressor.LevelPruner([{'sparsity':0.8,'support_type': 'default'}])
pruner(model_graph)
```

Pytorch code
```
pruner = nni.compressors.torchCompressor.LevelPruner([{'sparsity':0.8,'support_type': 'default'}])
pruner(model)
```

Our compressor will automatically insert mask into your model, and you can train your model with masks without changing your training code. You will get a compressed model when you finish your training.

You can get more information in Algorithm details

#### Configuration
We now provide an default format for our build-in algorithm, algorithm designer can follow this format and use our default configure parser

Following our default format, user can set configure in his code or a yaml file. And pass configure to compressor by init() or load_configure()


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
tfAGPruner:       
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
### For compression algorithm designer
We use the instrumentation method to insert a node or function after the corresponding position in the model.  And we provide interface for designer to design compression algorithm easily.

If you want to use mask to prune a model, you can use Pruner as base class. And design your mask in calc_mask() method.

Tensorflow code
```
class YourPruner(nni.compressors.tfCompressor.TfPruner):
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
class YourPruner(nni.compressors.torchCompressor.TorchPruner):
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
class YourPruner(nni.compressors.tfCompressor.TfQuantizer):
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
class YourPruner(nni.compressors.torchCompressor.TorchQuantizer):
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
class YourPruner(nni.compressors.torchCompressor.TorchQuantizer):
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
if an designer wants to update mask every step,  designer can implement step() or update_epoch() method in his code, and tell user to call when use it

