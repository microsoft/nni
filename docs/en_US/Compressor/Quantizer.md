Quantizer on NNI Compressor
===

## Naive Quantizer

We provide Naive Quantizer to quantizer weight to default 8 bits, you can use it to test quantize algorithm without any configure.

### Usage
tensorflow
```python
nni.compressors.tensorflow.NaiveQuantizer()(model_graph)
```
pytorch
```python
nni.compressors.torch.NaiveQuantizer()(model)
```

***

## QAT Quantizer
In [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf), authors Benoit Jacob and Skirmantas Kligys provide an algorithm to quantize the model with training.

>We propose an approach that simulates quantization effects in the forward pass of training. Backpropagation still happens as usual, and all weights and biases are stored in floating point so that they can be easily nudged by small amounts. The forward propagation pass however simulates quantized inference as it will happen in the inference engine, by implementing in floating-point arithmetic the rounding behavior of the quantization scheme
>* Weights are quantized before they are convolved with the input. If batch normalization (see [17]) is used for the layer, the batch normalization parameters are “folded into” the weights before quantization.
>* Activations are quantized at points where they would be during inference, e.g. after the activation function is applied to a convolutional or fully connected layer’s output, or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets.


### Usage
You can quantize your model to 8 bits with the code below before your training code.

Tensorflow code
```python
from nni.compressors.tensorflow import QAT_Quantizer
config_list = [{ 'q_bits': 8, 'op_types': ['default'] }]
quantizer = QAT_Quantizer(config_list)
quantizer(tf.get_default_graph())
```
PyTorch code
```python
from nni.compressors.torch import QAT_Quantizer
config_list = [{ 'q_bits': 8, 'op_types': ['default'] }]
quantizer = QAT_Quantizer(config_list)
quantizer(model)
```

You can view example for more information

#### User configuration for QAT Quantizer
* **q_bits:** This is to specify the q_bits operations to be quantized to


***

## DoReFa Quantizer
In [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160), authors Shuchang Zhou and Yuxin Wu provide an algorithm named DoReFa to quantize the weight, activation and gradients with training.

### Usage
To implement DoReFa Quantizer, you can add code below before your training code

Tensorflow code
```python
from nni.compressors.tensorflow import DoReFaQuantizer
config_list = [{ 'q_bits': 8, 'op_types': 'default' }]
quantizer = DoReFaQuantizer(config_list)
quantizer(tf.get_default_graph())
```
PyTorch code
```python
from nni.compressors.torch import DoReFaQuantizer
config_list = [{ 'q_bits': 8, 'op_types': 'default' }]
quantizer = DoReFaQuantizer(config_list)
quantizer(model)
```

You can view example for more information

#### User configuration for DoReFa Quantizer
* **q_bits:** This is to specify the q_bits operations to be quantized to
