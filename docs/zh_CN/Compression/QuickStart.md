# Tutorial for Model Compression

```eval_rst
.. contents::
```

In this tutorial, we use the [first section](#quick-start-to-compress-a-model) to quickly go through the usage of model compression on NNI. Then use the [second section](#detailed-usage-guide) to explain more details of the usage.

## Quick Start to Compress a Model

NNI provides very simple APIs for compressing a model. The compression includes pruning algorithms and quantization algorithms. The usage of them are the same, thus, here we use [slim pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#slim-pruner) as an example to show the usage.

### Write configuration

Write a configuration to specify the layers that you want to prune. The following configuration means pruning all the `BatchNorm2d`s to sparsity 0.7 while keeping other layers unpruned.

```python
configure_list = [{
    'sparsity': 0.7,
    'op_types': ['BatchNorm2d'],
}]
```

The specification of configuration can be found [here](#specification-of-config-list). Note that different pruners may have their own defined fields in configuration, for exmaple `start_epoch` in AGP pruner. Please refer to each pruner's [usage](./Pruner.md) for details, and adjust the configuration accordingly.

### Choose a compression algorithm

Choose a pruner to prune your model. First instantiate the chosen pruner with your model and configuration as arguments, then invoke `compress()` to compress your model.

```python
pruner = SlimPruner(model, configure_list)
model = pruner.compress()
```

Then, you can train your model using traditional training approach (e.g., SGD), pruning is applied transparently during the training. Some pruners prune once at the beginning, the following training can be seen as fine-tune. Some pruners prune your model iteratively, the masks are adjusted epoch by epoch during training.

### Export compression result

After training, you get accuracy of the pruned model. You can export model weights to a file, and the generated masks to a file as well. Exporting onnx model is also supported.

```python
pruner.export_model(model_path='pruned_vgg19_cifar10.pth', mask_path='mask_vgg19_cifar10.pth')
```

The complete code of model compression examples can be found [here](https://github.com/microsoft/nni/blob/master/examples/model_compress/model_prune_torch.py).

### Speed up the model

Masks do not provide real speedup of your model. The model should be speeded up based on the exported masks, thus, we provide an API to speed up your model as shown below. After invoking `apply_compression_results` on your model, your model becomes a smaller one with shorter inference latency.

```python
from nni.compression.torch import apply_compression_results
apply_compression_results(model, 'mask_vgg19_cifar10.pth')
```

Please refer to [here](ModelSpeedup.md) for detailed description.

## Detailed Usage Guide

The example code for users to apply model compression on a user model can be found below:

PyTorch code

```python
from nni.compression.torch import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()
```

Tensorflow code

```python
from nni.compression.tensorflow import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(tf.get_default_graph(), config_list)
pruner.compress()
```


You can use other compression algorithms in the package of `nni.compression`. The algorithms are implemented in both PyTorch and TensorFlow (partial support on TensorFlow), under `nni.compression.torch` and `nni.compression.tensorflow` respectively. You can refer to [Pruner](./Pruner.md) and [Quantizer](./Quantizer.md) for detail description of supported algorithms. Also if you want to use knowledge distillation, you can refer to [KDExample](../TrialExample/KDExample.md)

A compression algorithm is first instantiated with a `config_list` passed in. The specification of this `config_list` will be described later.

The function call `pruner.compress()` modifies user defined model (in Tensorflow the model can be obtained with `tf.get_default_graph()`, while in PyTorch the model is the defined model class), and the model is modified with masks inserted. Then when you run the model, the masks take effect. The masks can be adjusted at runtime by the algorithms.

*Note that, `pruner.compress` simply adds masks on model weights, it does not include fine tuning logic. If users want to fine tune the compressed model, they need to write the fine tune logic by themselves after `pruner.compress`.*

### Specification of `config_list`

Users can specify the configuration (i.e., `config_list`) for a compression algorithm. For example,when compressing a model, users may want to specify the sparsity ratio, to specify different ratios for different types of operations, to exclude certain types of operations, or to compress only a certain types of operations. For users to express these kinds of requirements, we define a configuration specification. It can be seen as a python `list` object, where each element is a `dict` object.

The `dict`s in the `list` are applied one by one, that is, the configurations in latter `dict` will overwrite the configurations in former ones on the operations that are within the scope of both of them.

There are different keys in a `dict`. Some of them are common keys supported by all the compression algorithms:

* __op_types__: This is to specify what types of operations to be compressed. 'default' means following the algorithm's default setting.
* __op_names__: This is to specify by name what operations to be compressed. If this field is omitted, operations will not be filtered by it.
* __exclude__: Default is False. If this field is True, it means the operations with specified types and names will be excluded from the compression.

Some other keys are often specific to a certain algorithms, users can refer to [pruning algorithms](./Pruner.md) and [quantization algorithms](./Quantizer.md) for the keys allowed by each algorithm.

A simple example of configuration is shown below:

```python
[
    {
        'sparsity': 0.8,
        'op_types': ['default']
    },
    {
        'sparsity': 0.6,
        'op_names': ['op_name1', 'op_name2']
    },
    {
        'exclude': True,
        'op_names': ['op_name3']
    }
]
```

It means following the algorithm's default setting for compressed operations with sparsity 0.8, but for `op_name1` and `op_name2` use sparsity 0.6, and do not compress `op_name3`.

#### Quantization specific keys

Besides the keys explained above, if you use quantization algorithms you need to specify more keys in `config_list`, which are explained below.

* __quant_types__ : list of string.

Type of quantization you want to apply, currently support 'weight', 'input', 'output'. 'weight' means applying quantization operation to the weight parameter of modules. 'input' means applying quantization operation to the input of module forward method. 'output' means applying quantization operation to the output of module forward method, which is often called as 'activation' in some papers.

* __quant_bits__ : int or dict of {str : int}

bits length of quantization, key is the quantization type, value is the quantization bits length, eg.
```
{
    quant_bits: {
        'weight': 8,
        'output': 4,
        },
}
```
when the value is int type, all quantization types share same bits length. eg.
```
{
    quant_bits: 8, # weight or output quantization are all 8 bits
}
```

The following example shows a more complete `config_list`, it uses `op_names` (or `op_types`) to specify the target layers along with the quantization bits for those layers.
```
configure_list = [{
        'quant_types': ['weight'],        
        'quant_bits': 8, 
        'op_names': ['conv1']
    }, {
        'quant_types': ['weight'],
        'quant_bits': 4,
        'quant_start_step': 0,
        'op_names': ['conv2']
    }, {
        'quant_types': ['weight'],
        'quant_bits': 3,
        'op_names': ['fc1']
        },
       {
        'quant_types': ['weight'],
        'quant_bits': 2,
        'op_names': ['fc2']
        }
]
```
In this example, 'op_names' is the name of layer and four layers will be quantized to different quant_bits.

### APIs for Updating Fine Tuning Status

Some compression algorithms use epochs to control the progress of compression (e.g. [AGP](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#agp-pruner)), and some algorithms need to do something after every minibatch. Therefore, we provide another two APIs for users to invoke: `pruner.update_epoch(epoch)` and `pruner.step()`.

`update_epoch` should be invoked in every epoch, while `step` should be invoked after each minibatch. Note that most algorithms do not require calling the two APIs. Please refer to each algorithm's document for details. For the algorithms that do not need them, calling them is allowed but has no effect.

### Export Compressed Model

You can easily export the compressed model using the following API if you are pruning your model, `state_dict` of the sparse model weights will be stored in `model.pth`, which can be loaded by `torch.load('model.pth')`. In this exported `model.pth`, the masked weights are zero.

```
pruner.export_model(model_path='model.pth')
```

`mask_dict` and pruned model in `onnx` format(`input_shape` need to be specified) can also be exported like this:

```python
pruner.export_model(model_path='model.pth', mask_path='mask.pth', onnx_path='model.onnx', input_shape=[1, 1, 28, 28])
```

If you want to really speed up the compressed model, please refer to [NNI model speedup](./ModelSpeedup.md) for details.
