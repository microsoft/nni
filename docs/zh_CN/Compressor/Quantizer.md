NNI Compressor 中的 Quantizer
===

## Naive Quantizer

Naive Quantizer 将 Quantizer 权重默认设置为 8 位，可用它来测试量化算法。

### 用法
Tensorflow
```python
nni.compressors.tensorflow.NaiveQuantizer()(model_graph)
```
PyTorch
```python
nni.compressors.torch.NaiveQuantizer()(model)
```

***

## QAT Quantizer
在 [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) 中，作者 Benoit Jacob 和 Skirmantas Kligys 提出了一种算法在训练中量化模型。
> 我们提出了一种方法，在训练的前向过程中模拟量化效果。 此方法不影响反向传播，所有权重和偏差都使用了浮点数保存，因此能很容易的进行量化。 The forward propagation pass however simulates quantized inference as it will happen in the inference engine, by implementing in floating-point arithmetic the rounding behavior of the quantization scheme * Weights are quantized before they are convolved with the input. If batch normalization (see [17]) is used for the layer, the batch normalization parameters are “folded into” the weights before quantization. * Activations are quantized at points where they would be during inference, e.g. after the activation function is applied to a convolutional or fully connected layer’s output, or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets.


### Usage
You can quantize your model to 8 bits with the code below before your training code.

Tensorflow code
```python
from nni.compressors.tensorflow import QAT_Quantizer
config_list = [{ 'q_bits': 8, 'op_types': 'default' }]
quantizer = QAT_Quantizer(config_list)
quantizer(tf.get_default_graph())
```
PyTorch code
```python
from nni.compressors.torch import QAT_Quantizer
config_list = [{ 'q_bits': 8, 'op_types': 'default' }]
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

#### User configuration for QAT Quantizer
* **q_bits:** This is to specify the q_bits operations to be quantized to
