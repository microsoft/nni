# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Union


class Granularity:
    """
    The base class for handling the pruning granularity. It is used to config the different granularity levels user-friendly.

    Parameters
    ----------
    dim
        The dimensions that corresponding to the under pruning weight dimensions in collected data.
        None means one-to-one correspondence between pruned dimensions and data, which equal to set `dim` as all data dimensions.
        Only these `dim` will be kept and other dimensions of the data will be reduced.

        Example:

        If you want to prune the Conv2d weight in filter level, and the weight size is (32, 16, 3, 3) [out-channel, in-channel, kernal-size-1, kernal-size-2].
        Then the under pruning dimensions is [0], which means you want to prune the filter or out-channel.

            Case 1: Directly collect the conv module weight as data to calculate the metric.
            Then the data has size (32, 16, 3, 3).
            Mention that the dimension 0 of the data is corresponding to the under pruning weight dimension 0.
            So in this case, `dim=0` will set in `__init__`.

            Case 2: Use the output of the conv module as data to calculate the metric.
            Then the data has size (batch_num, 32, feature_map_size_1, feature_map_size_2).
            Mention that the dimension 1 of the data is corresponding to the under pruning weight dimension 0.
            So in this case, `dim=1` will set in `__init__`.

        In both of these two case, the metric of this module has size (32,).
    block_sparse_size
        This used to describe the block size a metric value represented. By default, None means the block size is ones(len(dim)).
        Make sure len(dim) == len(block_sparse_size), and the block_sparse_size dimension position is corresponding to dim.

        Example:

        The under pruning weight size is (768, 768), and you want to apply a block sparse on dim=[0] with block size [64, 768],
        then you can set block_sparse_size=[64]. The final metric size is (12,).
    """
    def __init__(self, dim: Union[int, List] = None, sparse_on_dim_size: Union[int, List] = None):
        self.dim = [dim] if isinstance(dim, int) else list(dim)
        self.block_sparse_size = [sparse_on_dim_size] if isinstance(sparse_on_dim_size) else sparse_on_dim_size


class FineGrained(Granularity):
    """
    Initial this class and passing it to pruner means using fine-grained pruning.
    """
    def __init__(self):
        super().__init__(dim=None, sparse_on_dim_size=None)


class CoarseGrained(Granularity):
    """
    Initial this class and passing it to pruner means using coarse-grained pruning.

    Parameters
    ----------
    sparse_size
        The block sparse size on the weight matrix, None means cross all elements on the current dimension.
        Each block is considered as a whole whether it needs to be masked.
        E.g. the weight metrix size is [10, 20, 30], and sparse_size is [1, 2, 3],
        then the mask block on the weight is [i, 2 * j: 2 * j + 2, 3 * k: 3 * k + 3].
        Also the weight metrix size is [10, 20, 30], and sparse_size is [1, 2, None],
        then the mask block on the weight is [i, 2 * j: 2 * j + 2, :].
    """
    def __init__(self, sparse_size: Union[int, List]):
        sparse_size = [sparse_size] if isinstance(sparse_size, int) else list(sparse_size)
        dim = []
        sparse_on_dim_size = []
        for d, s in enumerate(sparse_size):
            if s is None:
                continue
            assert isinstance(s, int)
            if s > 1:
                dim.append(d)
                sparse_on_dim_size.append(s)
        super().__init__(dim=dim, sparse_on_dim_size=sparse_on_dim_size)


class ChannelGrained(Granularity):
    """
    Initial this class and passing it to pruner means using channel pruning.

    channel_dim
        Mark which dimension is the dimension of the channel that needs to be pruned.
        E.g. in Linear and Conv2d, if you want to prune output channels, you can set channel_dim=[0],
        if you want to prune input channels, you can set channel_dim=[1].
    """
    def __init__(self, channel_dim: Union[int, List] = 0):
        channel_dim = [channel_dim] if isinstance(channel_dim, int) else list(channel_dim)
        assert all([dim >= 0 for dim in channel_dim])
        super().__init__(dim=channel_dim, sparse_on_dim_size=None)
