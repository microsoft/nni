Quantizer on NNI Compressor
===

## Naive Quantizer

We provide Naive Quantizer to quantizer weight to default 8 bits, you can use it to test quantize algorithm without any configure.

### Usage
tensorflow
```python
nni.compressors.tensorflow.NaiveQuantizer(model_graph).compress()
```
pytorch
```python
nni.compressors.torch.NaiveQuantizer(model).compress()
```

***

## QAT Quantizer
In [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf), authors Benoit Jacob and Skirmantas Kligys provide an algorithm to quantize the model with training.

>We propose an approach that simulates quantization effects in the forward pass of training. Backpropagation still happens as usual, and all weights and biases are stored in floating point so that they can be easily nudged by small amounts. The forward propagation pass however simulates quantized inference as it will happen in the inference engine, by implementing in floating-point arithmetic the rounding behavior of the quantization scheme
>* Weights are quantized before they are convolved with the input. If batch normalization (see [17]) is used for the layer, the batch normalization parameters are “folded into” the weights before quantization.
>* Activations are quantized at points where they would be during inference, e.g. after the activation function is applied to a convolutional or fully connected layer’s output, or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets.


### Usage
You can quantize your model to 8 bits with the code below before your training code.

PyTorch code
```python
from nni.compressors.torch import QAT_Quantizer
model = Mnist()

config_list = [{
    'quant_types': ['weight'],
    'quant_bits': {
        'weight': 8,
    }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
    'op_types':['Conv2d', 'Linear']
}, {
    'quant_types': ['output'],
    'quant_bits': 8,
    'quant_start_step': 7000,
    'op_types':['ReLU6']
}]
quantizer = QAT_Quantizer(model, config_list)
quantizer.compress()
```

You can view example for more information

#### User configuration for QAT Quantizer
* **quant_types:** : list of string
type of quantization you want to apply, currently support 'weight', 'input', 'output'
* **quant_bits:** int or dict of {str : int}
bits length of quantization, key is the quantization type, value is the length, eg. {'weight', 8},
when the type is int, all quantization types share same bits length
* **quant_start_step:** int
disable quantization until model are run by certain number of steps, this allows the network to enter a more stable
state where activation quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0

### note
batch normalization folding is currently not supported.
***

## DoReFa Quantizer
In [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160), authors Shuchang Zhou and Yuxin Wu provide an algorithm named DoReFa to quantize the weight, activation and gradients with training.

### Usage
To implement DoReFa Quantizer, you can add code below before your training code

Tensorflow code
```python
from nni.compressors.tensorflow import DoReFaQuantizer
config_list = [{ 'q_bits': 8, 'op_types': 'default' }]
quantizer = DoReFaQuantizer(tf.get_default_graph(), config_list)
quantizer.compress()
```
PyTorch code
```python
from nni.compressors.torch import DoReFaQuantizer
config_list = [{ 'q_bits': 8, 'op_types': 'default' }]
quantizer = DoReFaQuantizer(model, config_list)
quantizer.compress()
```

You can view example for more information

#### User configuration for DoReFa Quantizer
* **q_bits:** This is to specify the q_bits operations to be quantized to
