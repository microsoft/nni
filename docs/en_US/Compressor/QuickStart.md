# Quick Start to Compress a Model

NNI provides very simple APIs for compressing a model. The compression includes pruning algorithms and quantization algorithms. The usage of them are the same, thus, here we use slim pruner as an example to show the usage.

## Write configuration

Write a configuration to specify the layers that you want to prune. The following configuration means pruning all the `BatchNorm2d`s to sparsity 0.7 while keeping other layers unpruned.

```python
configure_list = [{
    'sparsity': 0.7,
    'op_types': ['BatchNorm2d'],
}]
```

The specification of configuration can be found [here](Overview.md#user-configuration-for-a-compression-algorithm). Note that different pruners may have their own defined fields in configuration, for exmaple `start_epoch` in AGP pruner. Please refer to each pruner's [usage](Overview.md#supported-algorithms) for details, and adjust the configuration accordingly.

## Choose a compression algorithm

Choose a pruner to prune your model. First instantiate the chosen pruner with your model and configuration as arguments, then invoke `compress()` to compress your model.

```python
pruner = SlimPruner(model, configure_list)
model = pruner.compress()
```

Then, you can train your model using traditional training approach (e.g., SGD), pruning is applied transparently during the training. Some pruners prune once at the beginning, the following training can be seen as fine-tune. Some pruners prune your model iteratively, the masks are adjusted epoch by epoch during training.

## Export compression result

After training, you get accuracy of the pruned model. You can export model weights to a file, and the generated masks to a file as well. Exporting onnx model is also supported.

```python
pruner.export_model(model_path='pruned_vgg19_cifar10.pth', mask_path='mask_vgg19_cifar10.pth')
```

The complete code of model compression examples can be found [here](https://github.com/microsoft/nni/blob/master/examples/model_compress/model_prune_torch.py).

## Speed up the model

Masks do not provide real speedup of your model. The model should be speeded up based on the exported masks, thus, we provide an API to speed up your model as shown below. After invoking `apply_compression_results` on your model, your model becomes a smaller one with shorter inference latency.

```python
from nni.compression.torch import apply_compression_results
apply_compression_results(model, 'mask_vgg19_cifar10.pth')
```

Please refer to [here](ModelSpeedup.md) for detailed description.