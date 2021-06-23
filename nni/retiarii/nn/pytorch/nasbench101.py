from typing import Callable, List

import torch.nn as nn


class _NasBench101CellFixed(nn.Module):
    def __init__()


class NasBench101Cell(nn.Module):
    """
    Cell structure that is proposed in NAS-Bench-101 [nasbench101]_ .

    The space of this cell architecture consists of all possible directed acyclic graphs on no more than ``num_nodes`` nodes,
    where each possible node (other than IN and OUT) has one of ``op_candidates``, representing the corresponding operation.
    Edges connecting the nodes can be no more than ``num_edges``. 
    To align with the paper settings, two vertices specially labeled as operation IN and OUT, are also counted into
    ``num_nodes`` in our implementaion, the default value of ``num_nodes`` is 7 and ``num_edges`` is 9.

    Input of this cell should be of shape :math:`[N, C_{in}, *]`, while output should be `[N, C_{out}, *]`. The shape
    of each hidden nodes will be first automatically computed, depending on the cell structure. Each of the ``op_candidates``
    should be a callable that accepts computed ``num_features`` and returns a ``Module``. For example,

    .. code-block:: python

        def conv_bn_relu(num_features):
            return nn.Sequential(
                nn.Conv2d(num_features, num_features, 1),
                nn.BatchNorm2d(num_features),
                nn.ReLU()
            )

    The output of each node is the sum of its input node feed into its operation, except for the last node (output node),
    which is the concatenation of its input *hidden* nodes, adding the *IN* node (if IN and OUT are connected).

    When input tensor is added with any other tensor, there could be shape mismatch. Therefore, a projection transformation
    is needed to transform the input tensor. In paper, this is simply a Conv1x1 followed by BN and ReLU. The ``projection``
    parameters accepts ``in_features`` and ``out_features``, returns a ``Module``. This parameter has no default value,
    as we hold no assumption that users are dealing with images. An example for this parameter is,

    .. code-block:: python

        def projection_fn(in_features, out_features):
            return nn.Conv2d(in_features, out_features, 1)

    References
    ----------
    .. [nasbench101] Ying, Chris, et al. "Nas-bench-101: Towards reproducible neural architecture search."
        International Conference on Machine Learning. PMLR, 2019.
    """

    def __new__(cls, blocks: Union[Callable[[], nn.Module], List[Callable[[], nn.Module]], nn.Module, List[nn.Module]],
                depth: Union[int, Tuple[int, int]], label: Optional[str] = None):
        try:
            repeat = get_fixed_value(label)
            return nn.Sequential(*cls._replicate_and_instantiate(blocks, repeat))
        except AssertionError:
            return super().__new__(cls)

    def __init__(self, op_candidates: List[Callable[[int], nn.Module]],
                 in_features: int, out_features: int, projection: Callable[[int, int], nn.Module],
                 num_nodes: int = 5, num_edges: int = 9):
