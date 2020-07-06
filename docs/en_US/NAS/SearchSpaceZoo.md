# Search Space Zoo

## DartsCell

DartsCell is extract from [CNN model](./DARTS.md) designed in this repo. [Operations](#predefined-operations) connecting with nodes which contained in the cell strucure is fixed.

```eval_rst
..  autoclass:: nni.nas.pytorch.search_space_zoo.DartsCell
    :members:
```

### Example Code

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
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.ops.PoolBN
    ```
* Skip Connection

    There is no connection between the two nodes.
* DilConv3x3 / DilConv5x5

    Dilated Conv with `kernal_size=3` or `kernal_size=5` and `padding=1`
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.ops.DilConv
    ```
* SepConv3x3 / SepConv5x5

    Depthwise separable Conv with `kernal_size=3` or `kernal_size=5` and `padding=1`
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.ops.SepConv
    ```
