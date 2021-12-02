# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class Granularity:
    def __init__(self, dim=None, sparse_on_dim_size=None):
        self.dim = dim
        self.block_sparse_size = sparse_on_dim_size


class FineGrained(Granularity):
    def __init__(self):
        super().__init__(dim=None, sparse_on_dim_size=None)


class CoarseGrained(Granularity):
    """
    [1, None, None]
    """
    def __init__(self, sparse_size):
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
    def __init__(self, channel_dim):
        channel_dim = [channel_dim] if isinstance(channel_dim, int) else list(channel_dim)
        assert all([dim >= 0 for dim in channel_dim])
        super().__init__(dim=channel_dim, sparse_on_dim_size=None)
