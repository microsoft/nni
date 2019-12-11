# Compressor

我们很高兴的宣布，基于 NNI 的模型压缩工具发布了 Alpha 版本。该版本仍处于试验阶段，根据用户反馈会进行改进。 诚挚邀请您使用、反馈，或更多贡献。

NNI 提供了易于使用的工具包来帮助用户设计并使用压缩算法。 其使用了统一的接口来支持 TensorFlow 和 PyTorch。 只需要添加几行代码即可压缩模型。 NNI 中也内置了一些流程的模型压缩算法。 用户还可以通过 NNI 强大的自动调参功能来找到最好的压缩后的模型，详见[自动模型压缩](./AutoCompression.md)。 另外，用户还能使用 NNI 的接口，轻松定制新的压缩算法，详见[教程](#customize-new-compression-algorithms)。

## 支持的算法

We have provided several compression algorithms, including several pruning and quantization algorithms:

**Pruning**

| 名称                                              | 算法简介                                                                                                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [Level Pruner](./Pruner.md#level-pruner)        | 根据权重的绝对值，来按比例修剪权重。                                                                                                                     |
| [AGP Pruner](./Pruner.md#agp-pruner)            | 自动的逐步剪枝（是否剪枝的判断：基于对模型剪枝的效果）[参考论文](https://arxiv.org/abs/1710.01878)                                                                    |
| [L1Filter Pruner](./Pruner.md#l1filter-pruner)  | 剪除卷积层中最不重要的过滤器 (PRUNING FILTERS FOR EFFICIENT CONVNETS)[参考论文](https://arxiv.org/abs/1608.08710)                                        |
| [Slim Pruner](./Pruner.md#slim-pruner)          | 通过修剪 BN 层中的缩放因子来修剪卷积层中的通道 (Learning Efficient Convolutional Networks through Network Slimming)[参考论文](https://arxiv.org/abs/1708.06519) |
| [Lottery Ticket Pruner](./Pruner.md#agp-pruner) | "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" 提出的剪枝过程。 它会反复修剪模型。 [参考论文](https://arxiv.org/abs/1803.03635) |
| [FPGM Pruner](./Pruner.md#fpgm-pruner)          | Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration [参考论文](https://arxiv.org/pdf/1811.00250.pdf)   |

**Quantization**

| Name                                                | Brief Introduction of Algorithm                                                                                                                                                                                            |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Naive Quantizer](./Quantizer.md#naive-quantizer)   | Quantize weights to default 8 bits                                                                                                                                                                                         |
| [QAT Quantizer](./Quantizer.md#qat-quantizer)       | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. [Reference Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) |
| [DoReFa Quantizer](./Quantizer.md#dorefa-quantizer) | DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. [Reference Paper](https://arxiv.org/abs/1606.06160)                                                                           |

## 内置压缩算法的用法

We use a simple example to show how to modify your trial code in order to apply the compression algorithms. Let's say you want to prune all weight to 80% sparsity with Level Pruner, you can add the following three lines into your code before training your model ([here](https://github.com/microsoft/nni/tree/master/examples/model_compress) is complete code).

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


You can use other compression algorithms in the package of `nni.compression`. The algorithms are implemented in both PyTorch and Tensorflow, under `nni.compression.torch` and `nni.compression.tensorflow` respectively. You can refer to [Pruner](./Pruner.md) and [Quantizer](./Quantizer.md) for detail description of supported algorithms. Also if you want to use knowledge distillation, you can refer to [KDExample](../TrialExample/KDExample.md)

The function call `pruner.compress()` modifies user defined model (in Tensorflow the model can be obtained with `tf.get_default_graph()`, while in PyTorch the model is the defined model class), and the model is modified with masks inserted. Then when you run the model, the masks take effect. The masks can be adjusted at runtime by the algorithms.

When instantiate a compression algorithm, there is `config_list` passed in. We describe how to write this config below.

### 压缩算法中的用户配置

When compressing a model, users may want to specify the ratio for sparsity, to specify different ratios for different types of operations, to exclude certain types of operations, or to compress only a certain types of operations. For users to express these kinds of requirements, we define a configuration specification. It can be seen as a python `list` object, where each element is a `dict` object. In each `dict`, there are some keys commonly supported by NNI compression:

* __op_types__：指定要压缩的操作类型。 'default' 表示使用算法的默认设置。
* __op_names__：指定需要压缩的操作的名称。 如果没有设置此字段，操作符不会通过名称筛选。
* __exclude__：默认为 False。 如果此字段为 True，表示要通过类型和名称，将一些操作从压缩中排除。

There are also other keys in the `dict`, but they are specific for every compression algorithm. For example, some , some.

The `dict`s in the `list` are applied one by one, that is, the configurations in latter `dict` will overwrite the configurations in former ones on the operations that are within the scope of both of them.

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

It means following the algorithm's default setting for compressed operations with sparsity 0.8, but for `op_name1` and `op_name2` use sparsity 0.6, and please do not compress `op_name3`.

### 其它 API

Some compression algorithms use epochs to control the progress of compression (e.g. [AGP](./Pruner.md#agp-pruner)), and some algorithms need to do something after every minibatch. Therefore, we provide another two APIs for users to invoke. One is `update_epoch`, you can use it as follows:

Tensorflow code

```python
pruner.update_epoch(epoch, sess)
```

PyTorch code

```python
pruner.update_epoch(epoch)
```

The other is `step`, it can be called with `pruner.step()` after each minibatch. Note that not all algorithms need these two APIs, for those that do not need them, calling them is allowed but has no effect.

You can easily export the compressed model using the following API if you are pruning your model, `state_dict` of the sparse model weights will be stored in `model.pth`, which can be loaded by `torch.load('model.pth')`

```
pruner.export_model(model_path='model.pth')
```

`mask_dict` and pruned model in `onnx` format(`input_shape` need to be specified) can also be exported like this:

```python
pruner.export_model(model_path='model.pth', mask_path='mask.pth', onnx_path='model.onnx', input_shape=[1, 1, 28, 28])
```

## 定制新的压缩算法

To simplify writing a new compression algorithm, we design programming interfaces which are simple but flexible enough. There are interfaces for pruner and quantizer respectively.

### 剪枝算法

If you want to write a new pruning algorithm, you can write a class that inherits `nni.compression.tensorflow.Pruner` or `nni.compression.torch.Pruner` depending on which framework you use. Then, override the member functions with the logic of your algorithm.

```python
# TensorFlow 中定制 Pruner。
# PyTorch 的 Pruner，只需将
# nni.compression.tensorflow.Pruner 替换为
# nni.compression.torch.Pruner
class YourPruner(nni.compression.tensorflow.Pruner):
    def __init__(self, model, config_list):
        """
        建议使用 NNI 定义的规范来配置
        """
        super().__init__(model, config_list)

    def calc_mask(self, layer, config):
        """
        Pruner 需要重载此方法来为权重提供掩码
        掩码必须与权重有相同的形状和类型。
        将对权重执行 ``mul()`` 操作。
        此方法会挂载到模型的 ``forward()`` 方法上。

        Parameters
        ----------
        layer: LayerInfo
            为 ``layer`` 的权重计算掩码
        config: dict
            生成权重所需要的掩码
        """
        return your_mask

    #  PyTorch 版本不需要 sess 参数
    def update_epoch(self, epoch_num, sess):
        pass

    #  PyTorch 版本不需要 sess 参数
    def step(self, sess):
        """
        根据需要可基于 bind_model 方法中的模型或权重进行操作
        """
        pass
```

For the simplest algorithm, you only need to override `calc_mask`. It receives the to-be-compressed layers one by one along with their compression configuration. You generate the mask for this weight in this function and return. Then NNI applies the mask for you.

Some algorithms generate mask based on training progress, i.e., epoch number. We provide `update_epoch` for the pruner to be aware of the training progress. It should be called at the beginning of each epoch.

Some algorithms may want global information for generating masks, for example, all weights of the model (for statistic information). Your can use `self.bound_model` in the Pruner class for accessing weights. If you also need optimizer's information (for example in Pytorch), you could override `__init__` to receive more arguments such as model's optimizer. Then `step` can process or update the information according to the algorithm. You can refer to [source code of built-in algorithms](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/compressors) for example implementations.

### 量化算法

The interface for customizing quantization algorithm is similar to that of pruning algorithms. The only difference is that `calc_mask` is replaced with `quantize_weight`. `quantize_weight` directly returns the quantized weights rather than mask, because for quantization the quantized weights cannot be obtained by applying mask.

```python
# TensorFlow 中定制 Quantizer。
# PyTorch 的 Quantizer，只需将
# nni.compression.tensorflow.Quantizer 替换为
# nni.compression.torch.Quantizer
class YourQuantizer(nni.compression.tensorflow.Quantizer):
    def __init__(self, model, config_list):
        """
        建议使用 NNI 定义的规范来配置
        """
        super().__init__(model, config_list)

    def quantize_weight(self, weight, config, **kwargs):
        """
        quantize 需要重载此方法来为权重提供掩码
        此方法挂载于模型的 :meth:`forward`。

        Parameters
        ----------
        weight : Tensor
            要被量化的权重
        config : dict
            权重量化的配置
        """

        # 此处逻辑生成 `new_weight`

        return new_weight

    def quantize_output(self, output, config, **kwargs):
        """
        重载此方法输出量化
        此方法挂载于模型的 `:meth:`forward`。

        Parameters
        ----------
        output : Tensor
            需要被量化的输出
        config : dict
            输出量化的配置
        """

        # 实现生成 `new_output`

        return new_output

    def quantize_input(self, *inputs, config, **kwargs):
        """
        重载此方法量化输入
        此方法挂载于模型的 :meth:`forward`。

        Parameters
        ----------
        inputs : Tensor
            需要量化的输入
        config : dict
            输入量化用的配置
        """

        # 实现生成 `new_input`

        return new_input

    # Pytorch 版本不需要 sess 参数
    def update_epoch(self, epoch_num, sess):
        pass

    # Pytorch 版本不需要 sess 参数
    def step(self, sess):
        """
       根据需要可基于 bind_model 方法中的模型或权重进行操作
        """
        pass
```

### 使用用户自定义的压缩算法

__[TODO]__ ...
