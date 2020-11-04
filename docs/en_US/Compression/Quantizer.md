# Supported Quantization Algorithms on NNI

Index of supported quantization algorithms
* [Naive Quantizer](#naive-quantizer)
* [QAT Quantizer](#qat-quantizer)
* [DoReFa Quantizer](#dorefa-quantizer)
* [BNN Quantizer](#bnn-quantizer)

## Naive Quantizer

We provide Naive Quantizer to quantizer weight to default 8 bits, you can use it to test quantize algorithm without any configure.

### Usage
pytorch
```python 
model = nni.algorithms.compression.pytorch.quantization.NaiveQuantizer(model).compress()
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
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
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

common configuration needed by compression algorithms can be found at [Specification of `config_list`](./QuickStart.md).

configuration needed by this algorithm :

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

PyTorch code
```python
from nni.algorithms.compression.pytorch.quantization import DoReFaQuantizer
config_list = [{ 
    'quant_types': ['weight'],
    'quant_bits': 8, 
    'op_types': 'default' 
}]
quantizer = DoReFaQuantizer(model, config_list)
quantizer.compress()
```

You can view example for more information

#### User configuration for DoReFa Quantizer

common configuration needed by compression algorithms can be found at [Specification of `config_list`](./QuickStart.md).

configuration needed by this algorithm :

***

## BNN Quantizer

In [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830), 

>We introduce a method to train Binarized Neural Networks (BNNs) - neural networks with binary weights and activations at run-time. At training-time the binary weights and activations are used for computing the parameters gradients. During the forward pass, BNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations, which is expected to substantially improve power-efficiency.


### Usage

PyTorch code
```python
from nni.algorithms.compression.pytorch.quantization import BNNQuantizer
model = VGG_Cifar10(num_classes=10)

configure_list = [{
    'quant_bits': 1,
    'quant_types': ['weight'],
    'op_types': ['Conv2d', 'Linear'],
    'op_names': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'classifier.0', 'classifier.3']
}, {
    'quant_bits': 1,
    'quant_types': ['output'],
    'op_types': ['Hardtanh'],
    'op_names': ['features.6', 'features.9', 'features.13', 'features.16', 'features.20', 'classifier.2', 'classifier.5']
}]

quantizer = BNNQuantizer(model, configure_list)
model = quantizer.compress()
```

You can view example [examples/model_compress/BNN_quantizer_cifar10.py]( https://github.com/microsoft/nni/tree/v1.9/examples/model_compress/BNN_quantizer_cifar10.py) for more information.

#### User configuration for BNN Quantizer

common configuration needed by compression algorithms can be found at [Specification of `config_list`](./QuickStart.md).

configuration needed by this algorithm :

### Experiment

We implemented one of the experiments in [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830), we quantized the **VGGNet** for CIFAR-10 in the paper. Our experiments results are as follows:

| Model         | Accuracy  | 
| ------------- | --------- | 
| VGGNet        | 86.93%    |


The experiments code can be found at [examples/model_compress/BNN_quantizer_cifar10.py]( https://github.com/microsoft/nni/tree/v1.9/examples/model_compress/BNN_quantizer_cifar10.py) 
