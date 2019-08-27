Quantizer on NNI Compressor
===
## NaiveQuantizer

We provide NaiveQuantizer to quantizer weight to default 8 bits, you can use it to test quantize algorithm.

### Usage
tensorflow
```
nni.compressors.tf_compressor.NaiveQuantizer()(model_graph)
```
pytorch
```
nni.compressors.torch_compressor.NaiveQuantizer()(model)
```
***
## QATquantizer
In [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf), authors Benoit Jacob and Skirmantas Kligys provide an algorithm to quantize the model with training.

>We propose an approach that simulates quantization effects in the forward pass of training. Backpropagation still happens as usual, and all weights and biases are stored in floating point so that they can be easily nudged by small amounts. The forward propagation pass however simulates quantized inference as it will happen in the inference engine, by implementing in floating-point arithmetic the rounding behavior of the quantization scheme
>* Weights are quantized before they are convolved with the input. If batch normalization (see [17]) is used for the layer, the batch normalization parameters are “folded into” the weights before quantization.
>* Activations are quantized at points where they would be during inference, e.g. after the activation function is applied to a convolutional or fully connected layer’s output, or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets.



### Usage
You can quantize your model to 8 bits with the code below before your training code.

Tensorflow code
```
from nni.compressors.tfCompressor import QATquantizer
quantizer = QATquantizer(q_bits = 8)
quantizer(tf.get_default_graph())
```
Pytorch code
```
from nni.compressors.torchCompressor import QATquantizer
quantizer = QATquantizer(q_bits = 8)
quantizer(model)
```

You can view example for more information

***
## DoReFaQuantizer
In [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160), authors Shuchang Zhou and Yuxin Wu provide an algorithm named DoReFa to quantize the weight, activation and gradients with training.

### Usage
To implement DoReFaQuantizer, you can add code below before your training code

Tensorflow code
```
from nni.compressors.tfCompressor import DoReFaQuantizer
quantizer = DoReFaQuantizer(q_bits = 8)
quantizer(tf.get_default_graph())
```
Pytorch code
```
from nni.compressors.torchCompressor import DoReFaQuantizer
quantizer = DoReFaQuantizer(q_bits = 8)
quantizer(model)
```

You can view example for more information
