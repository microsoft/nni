# Search Space Zoo

## DartsCell

DartsCell is extracted from [CNN model](./DARTS.md) designed in this repo. [Operations](#darts-predefined-operations) connecting with nodes which contained in the cell strucure is fixed.

The predefined operations are shown as follows:

* MaxPool: call `torch.nn.MaxPool2d`. This operation applies a 2D max pooling over all input channels. Its parameters `kernal_size=3` and `padding=1` are fixed.
* AvgPool: call `torch.nn.AvgPool2d`. This operation applies a 2D average pooling over all input channels. Its parameters `kernal_size=3` and `padding=1` are fixed.
* Skip Connect: There is no operation between two nodes. Call `torch.nn.Identity` to forward what it gets to the output.
* SepConv3x3: Composed of two DilConvs with fixed `kernal_size=3` sequentially.
* SepConv5x5: Do the same operation as the previous one but it has different kernal size, which is set to 5.
* <a name="DilConv"></a>DilConv3x3:  (Dilated) depthwise separable Conv. It first calls `torch.nn.Conv2d` with fixed `kernal_size=3` to partition the feature map into `C_in` groups then applies 1x1 Convolution to get `C_out` output channels. It makes extracting features on every channel separately possible and reduces the number of parameters.
* DilConv5x5: Do the same operation as the previous one but it has different kernal size, which is set to 5.

```eval_rst
..  autoclass:: nni.nas.pytorch.search_space_zoo.DartsCell
    :members:
```

### Example Code

[example code](https://github.com/microsoft/nni/tree/master/examples/nas/search_space_zoo/darts_example.py)

```bash
git clone https://github.com/Microsoft/nni.git
cd nni/examples/nas/search_space_zoo
# search the best structure
python3 darts_example.py
```

<a class="predefined-operations-darts"></a>

### DARTS predefined operations

* MaxPool / AvgPool

    MaxPool / AvgPool with `kernal_size=3` and `padding=1` followed by BatchNorm2d
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.darts_ops.PoolBN
    ```
* Skip Connection

    There is no connection between the two nodes.
* DilConv3x3 / DilConv5x5

    Dilated Conv with `kernal_size=3` or `kernal_size=5` and `padding=1`
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.darts_ops.DilConv
    ```
* SepConv3x3 / SepConv5x5

    Depthwise separable Conv with `kernal_size=3` or `kernal_size=5` and `padding=1`
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.darts_ops.SepConv
    ```

## ENASMicroLayer

This layer is extracted from model designed [here](./ENAS.md). A model contains several blocks whose architecture keeps the same. A block is made up of some `ENAMicroLayer` 
and one `ENASReduceLayer`. The only difference between the two layers is that `ENASReduceLayer` applies all operations with `stride=2`.

An `ENASMicroLayer` contains `num_nodes` nodes and searches the topology among them. The first two nodes in a layer stand for the outputs from previous previous layer and previous layer respectively. The following nodes choose two previous nodes as input and apply two operations from [predefined ones](#predefined-operations-enas) then add them as the output of this node. For example, Node 4 chooses Node 1 and Node 3 as inputs then apply `MaxPool` and `AvgPool` on the inputs respectively. So the output of 
Node 4 is `MaxPool(Node 1)+AvgPool(Node 3)`. Nodes that are not served as input for other nodes are viewed as the output of the layer. If there are multiple output nodes, 
the model will concat them in channels as the layer output.

The predefined operations are listed as follows. Details can be seen [here](#predefined-operations-enas).

* MaxPool: call `torch.nn.MaxPool2d`. This operation applies a 2D max pooling over all input channels. Its parameters are fixed to `kernal_size=3`, `stride=1` and `padding=1`.
* AvgPool: call `torch.nn.AvgPool2d`. This operation applies a 2D average pooling over all input channels. Its parameters are fixed to `kernal_size=3`, `stride=1` and `padding=1`.
* SepConvBN3x3: ReLU followed by a [DilConv](#DilConv) and BatchNorm. Convilution parameters are `kernal_size=3`, `stride=1` and `padding=1`.
* SepConvBN5x5: Do the same operation as the previous one but it has different kernal size, which is set to 5.
* Skip Connect: There is no operation between two nodes. Call `torch.nn.Identity` to forward what it gets to the output.

```eval_rst
..  autoclass:: nni.nas.pytorch.search_space_zoo.ENASMicroLayer
    :members:
```

The Reduction Layer is made up by two Conv operations, each of them will output `C_out//2` channels and concat them in channels as the output. The Convolutions have `kernal_size=1` 
and `stride=2`, and they perform alternate sampling on the input so as to reduce the resolution without lossing information.

```eval_rst
..  autoclass:: nni.nas.pytorch.search_space_zoo.ENASReductionLayer
    :members:
```

### Example Code

[example code](https://github.com/microsoft/nni/tree/master/examples/nas/search_space_zoo/enas_micro_example.py)

```bash
git clone https://github.com/Microsoft/nni.git
cd nni/examples/nas/search_space_zoo
# search the best cell structure
python3 enas_micro_example.py
```

<a name="predefined-operations-enas"></a>

### ENAS Micro predefined operations

* MaxPool / AvgPool

    MaxPool / AvgPool with `kernal_size=3`, `stride=1` and `padding=1` followed by BatchNorm2d
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.enas_ops.Pool
    ```

* SepConv

    <!-- MaxPool / AvgPool with `kernal_size=3`, `stride=1` and `padding=1` followed by BatchNorm2d -->
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.enas_ops.SepConvBN
    ```

* Skip Connection

    There is no connection between the two nodes.

## ENASMacroLayer

In Macro search, the controller makes two decisions for each layer:L i) the [operation](#macro-operations) to perform on the previous layer, ii) the previous layer to connect to for skip connections. NNI privides [predefined operations](#macro-operations) for macro search, which are listed as following:

* Conv3x3(separable and non-separable): Conv parameters are fixed `kernal_size=3`, `padding=1` and `stride=1`. If `separable=True`, Conv is replaced with [DilConv](#DilConv).
* Conv5x5(separable and non-separable): Do the same operation as the previous one but it has different kernal size, which is set to 5.
* AvgPool: call `torch.nn.AvgPool2d`. This operation applies a 2D average pooling over all input channels. Its parameters are fixed to `kernal_size=3`, `stride=1` and `padding=1`.
* MaxPool: call `torch.nn.MaxPool2d`. This operation applies a 2D max pooling over all input channels. Its parameters are fixed to `kernal_size=3`, `stride=1` and `padding=1`.

```eval_rst
..  autoclass:: nni.nas.pytorch.search_space_zoo.ENASMacroLayer
    :members:
```

### Example Code

[example code](https://github.com/microsoft/nni/tree/master/examples/nas/search_space_zoo/enas_macro_example.py)

```bash
git clone https://github.com/Microsoft/nni.git
cd nni/examples/nas/search_space_zoo
# search the best cell structure
python3 enas_macro_example.py
```

<a name="macro-operations"></a>

### ENAS Macro predefined operations

* ConvBranch

    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.enas_ops.ConvBranch
    ```
* PoolBranch

    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.enas_ops.PoolBranch
    ```
