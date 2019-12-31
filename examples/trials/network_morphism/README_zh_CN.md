# 在 NNI 中用网络形态算法来进行自动模型结构搜索

Network Morphism （网络形态）是内置的 Tuner，它使用了网络形态技术来搜索和评估新的网络结构。 该示例展示了如何使用它来为深度学习找到好的模型架构。

## 如何运行此示例？

### 1. 训练框架支持

网络形态当前基于框架，还没有实现与框架脱离的方法。 当前支持 PyTorch 和 Keras。 如果熟悉 JSON 中间格式，可以在自定义的训练框架中生成自己的模型。 随后，我们会将中间结果从 JSON 转换为 ONNX，从而能够成为[标准的中间表示](https://github.com/onnx/onnx/blob/master/docs/IR.md)。

### 2. 安装需求

```bash
# 安装依赖包
cd examples/trials/network_morphism/
pip install -r requirements.txt
```

### 3. 更新配置

修改 `examples/trials/network_morphism/cifar10/config.yml` 来适配自己的任务。注意，searchSpacePath 在配置中不需要。 默认配置：

```yaml
authorName: default
experimentName: example_cifar10-network-morphism
trialConcurrency: 1
maxExecDuration: 48h
maxTrialNum: 200
#可选项: local, remote, pai
trainingServicePlatform: local
#可选项: true, false
useAnnotation: false
tuner:
  #可选项: TPE, Random, Anneal, Evolution, BatchTuner, NetworkMorphism
  #SMAC (SMAC 需要通过 nnictl 安装)
  builtinTunerName: NetworkMorphism
  classArgs:
    #可选项: maximize, minimize
    optimize_mode: maximize
    #当前仅支持视觉领域
    task: cv
    #修改来适配自己的图像大小
    input_width: 32
    #修改来适配自己的图像通道
    input_channel: 3
    #修改来适配自己的分类数量
    n_output_node: 10
trial:
  # 自己的命令
  command: python3 cifar10_keras.py
  codeDir: .
  gpuNum: 0
```

在 "trial" 部分中，如果需要使用 GPU 来进行架构搜索，可将 `gpuNum` 从 `0` 改为 `1`。 根据训练时长，可以增加 `maxTrialNum` 和 `maxExecDuration`。

`trialConcurrency` 是并发运行的 Trial 的数量。如果将 `gpuNum` 设置为 1，则需要与 GPU 数量一致。

### 4. 在代码中调用 "json\_to\_graph()" 函数

修改代码来调用 "json\_to\_graph()" 函数来从收到的 JSON 字符串生成一个 Pytorch 或 Keras 模型。 简单示例：

```python
import nni
from nni.networkmorphism_tuner.graph import json_to_graph

def build_graph_from_json(ir_model_json):
    """从 JSON 生成 Pytorch 模型
    """
    graph = json_to_graph(ir_model_json)
    model = graph.produce_torch_model()
    return model

# 从网络形态 Tuner 中获得下一组参数
RCV_CONFIG = nni.get_next_parameter()
# 调用函数来生成 Pytorch 或 Keras 模型
net = build_graph_from_json(RCV_CONFIG)

# 训练过程
# ....

# 将最终精度返回给 NNI
nni.report_final_result(best_acc)
```

### 5. 提交任务

```bash
# 可以使用命令行工具 "nnictl" 来创建任务
# 最终会成功提交一个网络形态任务到 NNI
nnictl create --config config.yml
```

## Trial 示例

下面的代码可在 `examples/trials/network_morphism/` 中找到。 可参考此代码来更新自己的任务。 希望它对你有用。

### FashionMNIST

`Fashion-MNIST` 是来自 [Zalando](https://jobs.zalando.com/tech/) 文章的图片 — 有 60,000 个示例的训练集和 10,000 个示例的测试集。 每个示例是 28x28 的灰度图，分为 10 个类别。 由于 MNIST 数据集过于简单，该数据集现在开始被广泛使用，用来替换 MNIST 作为基准数据集。

这里有两个示例，[FashionMNIST-keras.py](./FashionMNIST/FashionMNIST_keras.py) 和 [FashionMNIST-pytorch.py](./FashionMNIST/FashionMNIST_pytorch.py)。 注意，在 `config.yml` 中，需要为此数据集修改 `input_width` 为 28，以及 `input_channel` 为 1。

### Cifar10

`CIFAR-10` 数据集 [Canadian Institute For Advanced Research](https://www.cifar.ca/) 是广泛用于机器学习和视觉算法训练的数据集。 它是机器学习领域最广泛使用的数据集之一。 CIFAR-10 数据集包含了 60,000 张 32x32 的彩色图片，分为 10 类。

这里有两个示例，[cifar10-keras.py](./cifar10/cifar10_keras.py) 和 [cifar10-pytorch.py](./cifar10/cifar10_pytorch.py)。 在 `config.yml` 中，该数据集 `input_width` 的值是 32，并且 `input_channel` 是 3。