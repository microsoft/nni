# Network Morphism for Automatic Model Architecture Search in NNI
This example shows us how to use it to find good model architectures for deep learning. The NetworkMorphismTuner is a Tuner using network morphism techniques to generate the new network architecture.

## How to run this example?

### Training framework support

The network morphism now is framework-based, and we have not implemented the framework-free methods. The training frameworks which we have supported yet are Pytorch and Keras. If you get familiar with the intermediate JSON format, you can build your own model in your own training framework.



Here is an example of the build the model using Pytorch constructed from an instance of Graph.

```python
class TorchModel(torch.nn.Module):
    """A neural network class using pytorch constructed from an instance of Graph."""

    def __init__(self, graph):
        super(TorchModel, self).__init__()
        self.graph = graph
        self.layers = []
        for layer in graph.layer_list:
            self.layers.append(layer.to_real_layer())
        if graph.weighted:
            for index, layer in enumerate(self.layers):
                set_stub_weight_to_torch(self.graph.layer_list[index], layer)
        for index, layer in enumerate(self.layers):
            self.add_module(str(index), layer)

    def forward(self, input_tensor):
        topo_node_list = self.graph.topological_order
        output_id = topo_node_list[-1]
        input_id = topo_node_list[0]

        node_list = deepcopy(self.graph.node_list)
        node_list[input_id] = input_tensor

        for v in topo_node_list:
            for u, layer_id in self.graph.reverse_adj_list[v]:
                layer = self.graph.layer_list[layer_id]
                torch_layer = self.layers[layer_id]

                if isinstance(layer, (StubAdd, StubConcatenate)):
                    edge_input_tensor = list(
                        map(
                            lambda x: node_list[x],
                            self.graph.layer_id_to_input_node_ids[layer_id],
                        )
                    )
                else:
                    edge_input_tensor = node_list[u]

                temp_tensor = torch_layer(edge_input_tensor)
                node_list[v] = temp_tensor
        return node_list[output_id]

    def set_weight_to_graph(self):
        self.graph.weighted = True
        for index, layer in enumerate(self.layers):
            set_torch_weight_to_stub(layer, self.graph.layer_list[index])
```



Here is an example of the build the model using Keras constructed from an instance of Graph.

```python
class KerasModel:
    """A neural network class using keras constructed from an instance of Graph."""
    def __init__(self, graph):
        import keras

        self.graph = graph
        self.layers = []
        for layer in graph.layer_list:
            self.layers.append(to_real_keras_layer(layer))

        # Construct the keras graph.
        # Input
        topo_node_list = self.graph.topological_order
        output_id = topo_node_list[-1]
        input_id = topo_node_list[0]
        input_tensor = keras.layers.Input(shape=graph.node_list[input_id].shape)

        node_list = deepcopy(self.graph.node_list)
        node_list[input_id] = input_tensor

        # Output
        for v in topo_node_list:
            for u, layer_id in self.graph.reverse_adj_list[v]:
                layer = self.graph.layer_list[layer_id]
                keras_layer = self.layers[layer_id]

                if isinstance(layer, (StubAdd, StubConcatenate)):
                    edge_input_tensor = list(
                        map(
                            lambda x: node_list[x],
                            self.graph.layer_id_to_input_node_ids[layer_id],
                        )
                    )
                else:
                    edge_input_tensor = node_list[u]

                temp_tensor = keras_layer(edge_input_tensor)
                node_list[v] = temp_tensor

        output_tensor = node_list[output_id]
        self.model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

        if graph.weighted:
            for index, layer in enumerate(self.layers):
                set_stub_weight_to_keras(self.graph.layer_list[index], layer)

    def set_weight_to_graph(self):
        self.graph.weighted = True
        for index, layer in enumerate(self.layers):
            set_keras_weight_to_stub(layer, self.graph.layer_list[index])
```

As we can see, this Class above is actually converts the internal model DAG configuration `graph` to a Pytorch model or Keras model.

### Install the requirements 

```bash
# install the required packages
cd examples/trials/network-morphism/
pip install -r requirements.txt
```

### Update configuration

Modify `examples/trials/network-morphism/config.yaml`, here is the default configuration:

```bash
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
    #for now, we only support cv domain
    task: cv
    #input image width
    input_width: 32
    #input image channel
    input_channel: 3
    #output nums
    n_output_node: 10
trial:
  command: python cifar10-pytorch.py
  codeDir: .
  gpuNum: 0
```

In the "trial" part, if you want to use GPU to perform the architecture search, change `gpuNum` from `0` to `1`. You need to increase the `maxTrialNum` and `maxExecDuration`, according to how long you want to wait for the search result.

`trialConcurrency` is the number of trials running concurrently, which is the number of GPUs you want to use, if you are setting `gpuNum` to 1.

### Submit this job 

```bash
# You can use nni command tool "nnictl" to create the a job which submit to the nni
nnictl create --config config.yaml
```

## Technical details about the trial

The Bayesian network morphism based architecture for question answering has two different parts just like any other examples: the trial and the tuner.

### The trial

The trial has some examples which can guide you:

In those files, the graph to model part is important. 

```python
from nni.networkmorphism_tuner.graph import json_to_graph

def build_graph_from_json(ir_model_json):
    """build a pytorch model from json representation
    """
    graph = json_to_graph(ir_model_json)
    model = graph.produce_torch_model()
    return model
```

### The tuner

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

Here is its skeleton code about the tuner: 

```python
class NetworkMorphismTuner(Tuner):
    # ......

    def generate_parameters(self, parameter_id):
        """
        Returns a set of trial neural architecture, as a serializable object.
        parameter_id : int
        """
        if not self.history:
            self.init_search()

        new_father_id = None
        generated_graph = None
        if not self.training_queue:
            new_father_id, generated_graph = self.generate()
            new_model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((generated_graph, new_father_id, new_model_id))
            self.descriptors.append(generated_graph.extract_descriptor())

        graph, father_id, model_id = self.training_queue.pop(0)

        # from graph to json
        json_model_path = os.path.join(self.path, str(model_id) + ".json")
        json_out = graph_to_json(graph, json_model_path)
        self.total_data[parameter_id] = (json_out, father_id, model_id)

        return json_out
    
    def receive_trial_result(self, parameter_id, parameters, value):
        """ Record an observation of the objective function

        Arguments:           
            parameter_id : int
            parameters : dict of parameters
            value: final metrics of the trial, including reward     

        Raises:
            RuntimeError -- Received parameter_id not in total_data.
        """

        reward = self.extract_scalar_reward(value)

        if parameter_id not in self.total_data:
            raise RuntimeError("Received parameter_id not in total_data.")

        (_, father_id, model_id) = self.total_data[parameter_id]

        if self.optimize_mode is OptimizeMode.Maximize:
            reward = -reward

        graph = self.bo.searcher.load_model_by_id(model_id)

        # to use the value and graph
        self.add_model(value, graph, model_id)
        self.update(father_id, graph, value, model_id)
```



### The Network Representation Json Example

Here is an example of the intermediate representation JSON file we defined, which is passed from the tuner to the trial in the architecture search procedure. The example is as follows.

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

The definition of each model is a JSON object(also you can consider the model as a DAG graph), where:

- `input_shape` is a list of integers, which does not include the batch axis.
- `weighte` means whether the weights and biases in the neural network should be included in the graph.
- `operation_history` is the number of inputs the layer has.
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