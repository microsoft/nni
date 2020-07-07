# Search Space Zoo

## DartsCell

DartsCell is extract from [CNN model](./DARTS.md) designed in this repo. [Operations](#predefined-operations) connecting with nodes which contained in the cell strucure is fixed.

The predefined operations are shown as follows:

* MaxPool: call `torch.nn.MaxPool2d`. This operation applies a 2D max pooling over all input channels. Its parameters `kernal_size=3` and `padding=1` are fixed.
* AvgPool: call `torch.nn.AvgPool2d`. This operation applies a 2D average pooling over all input channels. Its parameters `kernal_size=3` and `padding=1` are fixed.
* Skip Connect: There is no operation between two nodes. Call `torch.nn.Identity` to forward what it gets to the output.
* SepConv3x3: Composed of two DilConvs with fixed `kernal_size=3` sequentially.
* SepConv5x5: Do the same operation as the previous one but it has different kernal size, which is set to 5.
* DilConv3x3:  (Dilated) depthwise separable conv. It first calls torch.nn.Conv2d with fixed `kernal_size=3` to partition the feature map into `C_in` groups then applies 1x1 Convolution to get `C_out` output channels. It makes extracting features on every channel seperately possible and reduces the number of parameters.
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

<a class="predefined-operations"></a>

### predefined operations

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
