# 在 NNI 中用网络形态算法来进行自动模型结构搜索

Network Morphism （网络形态）是内置的调参器，它使用了网络形态技术来搜索和评估新的网络结构。 该样例展示了如何使用它来为深度学习找到好的模型架构。

## 如何运行此样例？

### 1. 训练框架支持

网络形态当前基于框架，还没有实现与框架脱离的方法。 当前支持 Pytorch 和 Keras。 如果熟悉 JSON 中间格式，可以在自定义的训练框架中生成自己的模型。 随后，我们会将中间结果从 JSON 转换为 ONNX，从而能够成为[标准的中间表达](https://github.com/onnx/onnx/blob/master/docs/IR.md)。

### 2. 安装需求

```bash
# 安装依赖包
cd examples/trials/network_morphism/
pip install -r requirements.txt
```

### 3. 更新配置

Modify `examples/trials/network_morphism/cifar10/config.yaml` to fit your own task, note that searchSpacePath is not required in our configuration. Here is the default configuration:

```yaml
authorName: default
experimentName: example_cifar10-network-morphism
trialConcurrency: 1
maxExecDuration: 48h
maxTrialNum: 200
#choice: local, remote, pai
trainingServicePlatform: local
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, NetworkMorphism
  #SMAC (SMAC should be installed through nnictl) 
  builtinTunerName: NetworkMorphism
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
    #for now, this tuner only supports cv domain
    task: cv
    #modify to fit your input image width
    input_width: 32
    #modify to fit your input image channel
    input_channel: 3
    #modify to fit your number of classes
    n_output_node: 10
trial:
  # your own command here
  command: python3 cifar10_keras.py
  codeDir: .
  gpuNum: 0
```

In the "trial" part, if you want to use GPU to perform the architecture search, change `gpuNum` from `0` to `1`. You need to increase the `maxTrialNum` and `maxExecDuration`, according to how long you want to wait for the search result.

`trialConcurrency` is the number of trials running concurrently, which is the number of GPUs you want to use, if you are setting `gpuNum` to 1.

### 4. Call "json\_to\_graph()" function in your own code

Modify your code and call "json\_to\_graph()" function to build a pytorch model or keras model from received json string. Here is the simple example.

```python
import nni
from nni.networkmorphism_tuner.graph import json_to_graph

def build_graph_from_json(ir_model_json):
    """build a pytorch model from json representation
    """
    graph = json_to_graph(ir_model_json)
    model = graph.produce_torch_model()
    return model

# trial get next parameter from network morphism tuner
RCV_CONFIG = nni.get_next_parameter()
# call the function to build pytorch model or keras model
net = build_graph_from_json(RCV_CONFIG)

# training procedure
# ....

# report the final accuracy to nni
nni.report_final_result(best_acc)
```

### 5. Submit this job

```bash
# You can use nni command tool "nnictl" to create the a job which submit to the nni
# finally you successfully commit a Network Morphism Job to nni
nnictl create --config config.yaml
```

## Trial Examples

The trial has some examples which can guide you which located in `examples/trials/network_morphism/`. You can refer to it and modify to your own task. Hope this will help you to build your code.

### FashionMNIST

`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. It is a modern image classification dataset widely used to replacing MNIST as a baseline dataset, because the dataset MNIST is too easy and overused.

There are two examples, [FashionMNIST-keras.py](./FashionMNIST/FashionMNIST_keras.py) and [FashionMNIST-pytorch.py](./FashionMNIST/FashionMNIST_pytorch.py). Attention, you should change the `input_width` to 28 and `input_channel` to 1 in `config.yaml` for this dataset.

### Cifar10

The `CIFAR-10` dataset [Canadian Institute For Advanced Research](https://www.cifar.ca/) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.

There are two examples, [cifar10-keras.py](./cifar10/cifar10_keras.py) and [cifar10-pytorch.py](./cifar10/cifar10_pytorch.py). The value `input_width` is 32 and the value `input_channel` is 3 in `config.yaml` for this dataset.