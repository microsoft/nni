# Network Morphism Tuner on NNI

## 1. Introduction

[Autokeras](https://arxiv.org/abs/1806.10282) is a popular automl tools using Network Morphism. The basic idea of Autokeras is to use Bayesian Regression to estimate the metric of the Neural Network Architecture. Each time, it generates several child networks from father networks. Then it uses a naïve Bayesian regression estimate its metric value from history trained results of network and metric value pair. Next, it chooses the the child which has best estimated performance and adds it to the training queue. Inspired by its work and referring to its [code](https://github.com/jhfjhfj1/autokeras), we implement our Network Morphism method in our NNI platform.

If you want to know about network morphism trial usage, please check [Readme.md](../../../../../examples/trials/network-morphism/README.md) of the trial to get more detail.

## 2. Usage

To use Network Morphism, you should modify the following spec in your `config.yml` file:

```yaml
tuner:
  #choice: NetworkMorphism
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
```



In the training procedure, it generate a JSON file which represent a Network Graph. Users can call "json\_to\_graph()" function to build a pytorch model or keras model from this JSON file.

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

## 3. File Structure

The tuner has a lot of different files, functions and classes. Here we will only give most of those files a brief introduction:

- `networkmorphism_tuner.py` is a tuner which using network morphism techniques.

- `bayesian.py` is Bayesian method to estimate the metric of unseen model based on the models we have already searched.  
- `graph.py`  is the meta graph data structure. Class Graph is representing the neural architecture graph of a model.
  - Graph extracts the neural architecture graph from a model. 
  - Each node in the graph is a intermediate tensor between layers.
  - Each layer is an edge in the graph.
  - Notably, multiple edges may refer to the same layer.
- `graph_transformer.py` includes some graph transformer to wider, deeper or add a skip-connection into the graph.

- `layers.py`  includes all the layers we use in our model.
- `layer_transformer.py` includes some layer transformer to wider, deeper or add a skip-connection into the layer.
- `nn.py` includes the class to generate network class initially.
- `metric.py` some metric classes including Accuracy and MSE.
- `utils.py` is the example search network architectures in dataset `cifar10` by using Keras.

## 4. The Network Representation Json Example

Here is an example of the intermediate representation JSON file we defined, which is passed from the tuner to the trial in the architecture search procedure. Users can call "json\_to\_graph()" function in trial code to build a pytorch model or keras model from this JSON file. The example is as follows.

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

The definition of each model is a JSON object(also you can consider the model as a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)), where:

- `input_shape` is a list of integers, which does not include the batch axis.
- `weighted` means whether the weights and biases in the neural network should be included in the graph.
- `operation_history` is a list saving all the network morphism operations.
- `layer_id_to_input_node_ids` is a dictionary instance mapping from layer identifiers to their input nodes identifiers.
- `layer_id_to_output_node_ids` is a dictionary instance mapping from layer identifiers to their output nodes identifiers
- `adj_list` is a two dimensional list. The adjacency list of the graph. The first dimension is identified by tensor identifiers. In each edge list, the elements are two-element tuples of (tensor identifier, layer identifier).
- `reverse_adj_list` is a  A reverse adjacent list in the same format as adj_list.
- `node_list` is a list of integers. The indices of the list are the identifiers.
- `layer_list` is a list of stub layers. The indices of the list are the identifiers.
  - For `StubConv (StubConv1d, StubConv2d, StubConv3d)`, the number follows is its node input id(or id list), node output id, input_channel, filters, kernel_size, stride and padding.

  - For `StubDense`, the number follows is its node input id(or id list), node output id, input_units and units.

  - For `StubBatchNormalization (StubBatchNormalization1d, StubBatchNormalization2d, StubBatchNormalization3d)`, the number follows is its node input id(or id list), node output id and features numbers.

  - For `StubDropout(StubDropout1d, StubDropout2d, StubDropout3d)`, the number follows is its node input id(or id list), node output id and dropout rate.

  - For `StubPooling (StubPooling1d, StubPooling2d, StubPooling3d)`, the number follows is its node input id(or id list), node output id, kernel_size, stride and padding.

  - For else layers, the number follows is its node input id(or id list) and node output id.

## 5. TODO

Next step, we will change the API from fixed network generator to more network operator generator. Besides, we will use ONNX instead of JSON later as the intermediate representation spec in the future.
