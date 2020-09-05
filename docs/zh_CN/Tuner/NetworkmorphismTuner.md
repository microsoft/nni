# Network Morphism Tuner

## 1. 介绍

[Autokeras](https://arxiv.org/abs/1806.10282) 是使用 Network Morphism 算法的流行的自动机器学习工具。 Autokeras 的基本理念是使用贝叶斯回归来预测神经网络架构的指标。 每次都会从父网络生成几个子网络。 然后使用朴素贝叶斯回归，从网络的历史训练结果来预测它的指标值。 接下来，会选择预测结果最好的子网络加入训练队列中。 在[此代码](https://github.com/jhfjhfj1/autokeras)的启发下，我们在 NNI 中实现了 Network Morphism 算法。

要了解 Network Morphism Trial 的用法，参考 [Readme_zh_CN.md](https://github.com/Microsoft/nni/blob/master/examples/trials/network_morphism/README_zh_CN.md)。

## 2. 用法

要使用 Network Morphism，需要如下配置 `config.yml` 文件：

```yaml
tuner:
  #选择: NetworkMorphism
  builtinTunerName: NetworkMorphism
  classArgs:
    #可选项: maximize, minimize
    optimize_mode: maximize
    #当前仅支持 cv 领域
    task: cv
    #修改来支持实际图像宽度
    input_width: 32
    #修改来支持实际图像通道
    input_channel: 3
    #修改来支持实际的分类数量
    n_output_node: 10
```

在训练过程中，会生成一个 JSON 文件来表示网络图。 可调用 "json\_to\_graph()" 函数来将 JSON 文件转化为 Pytoch 或 Keras 模型。

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

如果需要保存并读取**最佳模型**，推荐采用以下方法。

```python
# 1. 使用 NNI API
## 可以通过 Web 界面获取最佳模型
## 或者通过 `nni-experiments/experiment_id/log/model_path/best_model.txt'

## 从模型文件中读取 json 字符串并使用 NNI API 加载
with open("best-model.json") as json_file:
    json_of_model = json_file.read()
model = build_graph_from_json(json_of_model)

# 2. 使用框架的 API (与具体框架相关)
## 2.1 Keras API

## 在 Trial 代码中使用 Keras API 保存
## 最好保存 NNI 的 ID
model_id = nni.get_sequence_id()
## 将模型序列化为 JSON
model_json = model.to_json()
with open("model-{}.json".format(model_id), "w") as json_file:
    json_file.write(model_json)
## 将权重序列化至 HDF5
model.save_weights("model-{}.h5".format(model_id))

## 重用模型时，使用 Keras API 读取
## 读取 JSON 文件，并创建模型
model_id = "" # 需要重用的模型 ID
with open('model-{}.json'.format(model_id), 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
## 将权重加载到新模型中
loaded_model.load_weights("model-{}.h5".format(model_id))

## 2.2 PyTorch API

## 在 Trial 代码中使用 PyTorch API 保存
model_id = nni.get_sequence_id()
torch.save(model, "model-{}.pt".format(model_id))

## 重用模型时，使用 PyTorch API 读取
model_id = "" # 需要重用的模型 ID
loaded_model = torch.load("model-{}.pt".format(model_id))

```

## 3. 文件结构

Tuner 有大量的文件、函数和类。 这里简单介绍最重要的文件：

- `networkmorphism_tuner.py` 是使用 network morphism 算法的 Tuner。

- `bayesian.py` 是用来基于已经搜索到的模型来预测未知模型指标的贝叶斯算法。

- `graph.py` 是元图数据结构。 类 Graph 表示了模型的神经网络图。 
  - Graph 从模型中抽取神经网络。
  - 图中的每个节点都是层之间的中间张量。
  - 在图中，边表示层。
  - 注意，多条边可能会表示同一层。

- `graph_transformer.py` 包含了一些图转换，包括变宽，变深，或在图中增加跳跃连接。

- `layers.py` 包括模型中用到的所有层。

- `layer_transformer.py` 包含了一些层转换，包括变宽，变深，或在层中增加跳跃连接。
- `nn.py` 包括生成初始网络的类。
- `metric.py` 包括了一些指标类，如 Accuracy 和 MSE。
- `utils.py` 是使用 Keras 在数据集 `cifar10` 上搜索神经网络的示例。

## 4. 网络表示的 JSON 示例

这是定义的中间表示 JSON 示例，在架构搜索过程中会从 Tuner 传到 Trial。 可调用 Trial 代码中的 "json\_to\_graph()" 函数来将 JSON 文件转化为 Pytoch 或 Keras 模型。

```json
{
     "input_shape": [32, 32, 3],
     "weighted": false,
     "operation_history": [],
     "layer_id_to_input_node_ids": {"0": [0],"1": [1],"2": [2],"3": [3],"4": [4],"5": [5],"6": [6],"7": [7],"8": [8],"9": [9],"10": [10],"11": [11],"12": [12],"13": [13],"14": [14],"15": [15],"16": [16]
     },
     "layer_id_to_output_node_ids": {"0": [1],"1": [2],"2": [3],"3": [4],"4": [5],"5": [6],"6": [7],"7": [8],"8": [9],"9": [10],"10": [11],"11": [12],"12": [13],"13": [14],"14": [15],"15": [16],"16": [17]
     },
     "adj_list": {
         "0": [[1, 0]],
         "1": [[2, 1]],
         "2": [[3, 2]],
         "3": [[4, 3]],
         "4": [[5, 4]],
         "5": [[6, 5]],
         "6": [[7, 6]],
         "7": [[8, 7]],
         "8": [[9, 8]],
         "9": [[10, 9]],
         "10": [[11, 10]],
         "11": [[12, 11]],
         "12": [[13, 12]],
         "13": [[14, 13]],
         "14": [[15, 14]],
         "15": [[16, 15]],
         "16": [[17, 16]],
         "17": []
     },
     "reverse_adj_list": {
         "0": [],
         "1": [[0, 0]],
         "2": [[1, 1]],
         "3": [[2, 2]],
         "4": [[3, 3]],
         "5": [[4, 4]],
         "6": [[5, 5]],
         "7": [[6, 6]],
         "8": [[7, 7]],
         "9": [[8, 8]],
         "10": [[9, 9]],
         "11": [[10, 10]],
         "12": [[11, 11]],
         "13": [[12, 12]],
         "14": [[13, 13]],
         "15": [[14, 14]],
         "16": [[15, 15]],
         "17": [[16, 16]]
     },
     "node_list": [
         [0, [32, 32, 3]],
         [1, [32, 32, 3]],
         [2, [32, 32, 64]],
         [3, [32, 32, 64]],
         [4, [16, 16, 64]],
         [5, [16, 16, 64]],
         [6, [16, 16, 64]],
         [7, [16, 16, 64]],
         [8, [8, 8, 64]],
         [9, [8, 8, 64]],
         [10, [8, 8, 64]],
         [11, [8, 8, 64]],
         [12, [4, 4, 64]],
         [13, [64]],
         [14, [64]],
         [15, [64]],
         [16, [64]],
         [17, [10]]
     ],
     "layer_list": [
         [0, ["StubReLU", 0, 1]],
         [1, ["StubConv2d", 1, 2, 3, 64, 3]],
         [2, ["StubBatchNormalization2d", 2, 3, 64]],
         [3, ["StubPooling2d", 3, 4, 2, 2, 0]],
         [4, ["StubReLU", 4, 5]],
         [5, ["StubConv2d", 5, 6, 64, 64, 3]],
         [6, ["StubBatchNormalization2d", 6, 7, 64]],
         [7, ["StubPooling2d", 7, 8, 2, 2, 0]],
         [8, ["StubReLU", 8, 9]],
         [9, ["StubConv2d", 9, 10, 64, 64, 3]],
         [10, ["StubBatchNormalization2d", 10, 11, 64]],
         [11, ["StubPooling2d", 11, 12, 2, 2, 0]],
         [12, ["StubGlobalPooling2d", 12, 13]],
         [13, ["StubDropout2d", 13, 14, 0.25]],
         [14, ["StubDense", 14, 15, 64, 64]],
         [15, ["StubReLU", 15, 16]],
         [16, ["StubDense", 16, 17, 64, 10]]
     ]
 }
```

可将模型视为[有向无环图](https://en.wikipedia.org/wiki/Directed_acyclic_graph)。 每个模型的定义都是一个 JSON 对象：

- `input_shape` 是整数的列表，不包括批量维度。
- `weighted` 表示是否权重和偏移值应该包含在此神经网络图中。
- `operation_history` 是保存了所有网络形态操作的列表。
- `layer_id_to_input_node_ids` 是字典，将层的标识映射到输入节点标识。
- `layer_id_to_output_node_ids` 是字典，将层的标识映射到输出节点标识。
- `adj_list` 是二维列表，是图的邻接表。 第一维是张量标识。 在每条边的列表中，元素是两元组（张量标识，层标识）。
- `reverse_adj_list` 是与 adj_list 格式一样的反向邻接列表。
- `node_list` 是一个整数列表。 列表的索引是标识。
- `layer_list` 是层的列表。 列表的索引是标识。
  
  - 对于 `StubConv(StubConv1d, StubConv2d, StubConv3d)`，后面的数字表示节点的输入 id（或 id 列表），节点输出 id，input_channel，filters，kernel_size，stride 和 padding。
  
  - 对于 `StubDense`，后面的数字表示节点的输入 id （或 id 列表），节点输出 id，input_units 和 units。
  
  - 对于 `StubBatchNormalization (StubBatchNormalization1d, StubBatchNormalization2d, StubBatchNormalization3d)`，后面的数字表示节点输入 id（或 id 列表），节点输出 id，和特征数量。
  
  - 对于 `StubDropout(StubDropout1d, StubDropout2d, StubDropout3d)`，后面的数字表示节点的输入 id （或 id 列表），节点的输出 id 和 dropout 率。
  
  - 对于 `StubPooling (StubPooling1d, StubPooling2d, StubPooling3d)`后面的数字表示节点的输入 id（或 id 列表），节点输出 id，kernel_size, stride 和 padding。
  
  - 对于其它层，后面的数字表示节点的输入 id（或 id 列表）以及节点的输出 id。

## 5. TODO

下一步，会将 API 从固定网络生成器，改为有更多可用操作的网络生成器。 会使用 ONNX 格式来替代 JSON 作为中间表示结果。