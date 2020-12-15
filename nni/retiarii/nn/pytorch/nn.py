import inspect
import logging
from typing import Any, List

import torch
import torch.nn as nn

from ...utils import add_record

_logger = logging.getLogger(__name__)

__all__ = [
    'LayerChoice', 'InputChoice', 'Placeholder',
    'Module', 'Sequential', 'ModuleList',  # TODO: 'ModuleDict', 'ParameterList', 'ParameterDict',
    'Identity', 'Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d',
    'ConvTranspose2d', 'ConvTranspose3d', 'Threshold', 'ReLU', 'Hardtanh', 'ReLU6',
    'Sigmoid', 'Tanh', 'Softmax', 'Softmax2d', 'LogSoftmax', 'ELU', 'SELU', 'CELU', 'GLU', 'GELU', 'Hardshrink',
    'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU', 'Softsign', 'Softmin',
    'Tanhshrink', 'RReLU', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'MaxPool1d', 'MaxPool2d',
    'MaxPool3d', 'MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d', 'FractionalMaxPool2d', "FractionalMaxPool3d",
    'LPPool1d', 'LPPool2d', 'LocalResponseNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'InstanceNorm1d',
    'InstanceNorm2d', 'InstanceNorm3d', 'LayerNorm', 'GroupNorm', 'SyncBatchNorm',
    'Dropout', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout',
    'ReflectionPad1d', 'ReflectionPad2d', 'ReplicationPad2d', 'ReplicationPad1d', 'ReplicationPad3d',
    'CrossMapLRN2d', 'Embedding', 'EmbeddingBag', 'RNNBase', 'RNN', 'LSTM', 'GRU', 'RNNCellBase', 'RNNCell',
    'LSTMCell', 'GRUCell', 'PixelShuffle', 'Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d',
    'PairwiseDistance', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d', 'AdaptiveAvgPool1d',
    'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d', 'TripletMarginLoss', 'ZeroPad2d', 'ConstantPad1d', 'ConstantPad2d',
    'ConstantPad3d', 'Bilinear', 'CosineSimilarity', 'Unfold', 'Fold',
    'AdaptiveLogSoftmaxWithLoss', 'TransformerEncoder', 'TransformerDecoder',
    'TransformerEncoderLayer', 'TransformerDecoderLayer', 'Transformer',
    #'LazyLinear', 'LazyConv1d', 'LazyConv2d', 'LazyConv3d',
    #'LazyConvTranspose1d', 'LazyConvTranspose2d', 'LazyConvTranspose3d',
    #'Unflatten', 'SiLU', 'TripletMarginWithDistanceLoss', 'ChannelShuffle',
    'Flatten', 'Hardsigmoid', 'Hardswish'
]


class LayerChoice(nn.Module):
    def __init__(self, op_candidates, reduction=None, return_mask=False, key=None):
        super(LayerChoice, self).__init__()
        self.candidate_ops = op_candidates
        self.label = key
        if reduction or return_mask:
            _logger.warning('input arguments `reduction` and `return_mask` are deprecated!')

    def forward(self, x):
        return x


class InputChoice(nn.Module):
    def __init__(self, n_candidates=None, choose_from=None, n_chosen=1,
                 reduction="sum", return_mask=False, key=None):
        super(InputChoice, self).__init__()
        self.n_chosen = n_chosen
        self.reduction = reduction
        self.label = key
        if n_candidates or choose_from or return_mask:
            _logger.warning('input arguments `n_candidates`, `choose_from` and `return_mask` are deprecated!')

    def forward(self, candidate_inputs: List[torch.Tensor]) -> torch.Tensor:
        # fake return
        return torch.tensor(candidate_inputs)  # pylint: disable=not-callable


class ValueChoice:
    """
    The instance of this class can only be used as input argument,
    when instantiating a pytorch module.
    TODO: can also be used in training approach
    """

    def __init__(self, candidate_values: List[Any]):
        self.candidate_values = candidate_values


class Placeholder(nn.Module):
    def __init__(self, label, related_info):
        add_record(id(self), related_info)
        self.label = label
        self.related_info = related_info
        super(Placeholder, self).__init__()

    def forward(self, x):
        return x


class ChosenInputs(nn.Module):
    def __init__(self, chosen: int):
        super().__init__()
        self.chosen = chosen

    def forward(self, candidate_inputs):
        # TODO: support multiple chosen inputs
        return candidate_inputs[self.chosen]

# the following are pytorch modules


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()


class Sequential(nn.Sequential):
    def __init__(self, *args):
        add_record(id(self), {})
        super(Sequential, self).__init__(*args)


class ModuleList(nn.ModuleList):
    def __init__(self, *args):
        add_record(id(self), {})
        super(ModuleList, self).__init__(*args)


def wrap_module(original_class):
    orig_init = original_class.__init__
    argname_list = list(inspect.signature(original_class).parameters.keys())
    # Make copy of original __init__, so we can call it without recursion

    def __init__(self, *args, **kws):
        full_args = {}
        full_args.update(kws)
        for i, arg in enumerate(args):
            full_args[argname_list[i]] = arg
        add_record(id(self), full_args)

        orig_init(self, *args, **kws)  # Call the original __init__

    original_class.__init__ = __init__  # Set the class' __init__ to the new one
    return original_class


# TODO: support different versions of pytorch
Identity = wrap_module(nn.Identity)
Linear = wrap_module(nn.Linear)
Conv1d = wrap_module(nn.Conv1d)
Conv2d = wrap_module(nn.Conv2d)
Conv3d = wrap_module(nn.Conv3d)
ConvTranspose1d = wrap_module(nn.ConvTranspose1d)
ConvTranspose2d = wrap_module(nn.ConvTranspose2d)
ConvTranspose3d = wrap_module(nn.ConvTranspose3d)
Threshold = wrap_module(nn.Threshold)
ReLU = wrap_module(nn.ReLU)
Hardtanh = wrap_module(nn.Hardtanh)
ReLU6 = wrap_module(nn.ReLU6)
Sigmoid = wrap_module(nn.Sigmoid)
Tanh = wrap_module(nn.Tanh)
Softmax = wrap_module(nn.Softmax)
Softmax2d = wrap_module(nn.Softmax2d)
LogSoftmax = wrap_module(nn.LogSoftmax)
ELU = wrap_module(nn.ELU)
SELU = wrap_module(nn.SELU)
CELU = wrap_module(nn.CELU)
GLU = wrap_module(nn.GLU)
GELU = wrap_module(nn.GELU)
Hardshrink = wrap_module(nn.Hardshrink)
LeakyReLU = wrap_module(nn.LeakyReLU)
LogSigmoid = wrap_module(nn.LogSigmoid)
Softplus = wrap_module(nn.Softplus)
Softshrink = wrap_module(nn.Softshrink)
MultiheadAttention = wrap_module(nn.MultiheadAttention)
PReLU = wrap_module(nn.PReLU)
Softsign = wrap_module(nn.Softsign)
Softmin = wrap_module(nn.Softmin)
Tanhshrink = wrap_module(nn.Tanhshrink)
RReLU = wrap_module(nn.RReLU)
AvgPool1d = wrap_module(nn.AvgPool1d)
AvgPool2d = wrap_module(nn.AvgPool2d)
AvgPool3d = wrap_module(nn.AvgPool3d)
MaxPool1d = wrap_module(nn.MaxPool1d)
MaxPool2d = wrap_module(nn.MaxPool2d)
MaxPool3d = wrap_module(nn.MaxPool3d)
MaxUnpool1d = wrap_module(nn.MaxUnpool1d)
MaxUnpool2d = wrap_module(nn.MaxUnpool2d)
MaxUnpool3d = wrap_module(nn.MaxUnpool3d)
FractionalMaxPool2d = wrap_module(nn.FractionalMaxPool2d)
FractionalMaxPool3d = wrap_module(nn.FractionalMaxPool3d)
LPPool1d = wrap_module(nn.LPPool1d)
LPPool2d = wrap_module(nn.LPPool2d)
LocalResponseNorm = wrap_module(nn.LocalResponseNorm)
BatchNorm1d = wrap_module(nn.BatchNorm1d)
BatchNorm2d = wrap_module(nn.BatchNorm2d)
BatchNorm3d = wrap_module(nn.BatchNorm3d)
InstanceNorm1d = wrap_module(nn.InstanceNorm1d)
InstanceNorm2d = wrap_module(nn.InstanceNorm2d)
InstanceNorm3d = wrap_module(nn.InstanceNorm3d)
LayerNorm = wrap_module(nn.LayerNorm)
GroupNorm = wrap_module(nn.GroupNorm)
SyncBatchNorm = wrap_module(nn.SyncBatchNorm)
Dropout = wrap_module(nn.Dropout)
Dropout2d = wrap_module(nn.Dropout2d)
Dropout3d = wrap_module(nn.Dropout3d)
AlphaDropout = wrap_module(nn.AlphaDropout)
FeatureAlphaDropout = wrap_module(nn.FeatureAlphaDropout)
ReflectionPad1d = wrap_module(nn.ReflectionPad1d)
ReflectionPad2d = wrap_module(nn.ReflectionPad2d)
ReplicationPad2d = wrap_module(nn.ReplicationPad2d)
ReplicationPad1d = wrap_module(nn.ReplicationPad1d)
ReplicationPad3d = wrap_module(nn.ReplicationPad3d)
CrossMapLRN2d = wrap_module(nn.CrossMapLRN2d)
Embedding = wrap_module(nn.Embedding)
EmbeddingBag = wrap_module(nn.EmbeddingBag)
RNNBase = wrap_module(nn.RNNBase)
RNN = wrap_module(nn.RNN)
LSTM = wrap_module(nn.LSTM)
GRU = wrap_module(nn.GRU)
RNNCellBase = wrap_module(nn.RNNCellBase)
RNNCell = wrap_module(nn.RNNCell)
LSTMCell = wrap_module(nn.LSTMCell)
GRUCell = wrap_module(nn.GRUCell)
PixelShuffle = wrap_module(nn.PixelShuffle)
Upsample = wrap_module(nn.Upsample)
UpsamplingNearest2d = wrap_module(nn.UpsamplingNearest2d)
UpsamplingBilinear2d = wrap_module(nn.UpsamplingBilinear2d)
PairwiseDistance = wrap_module(nn.PairwiseDistance)
AdaptiveMaxPool1d = wrap_module(nn.AdaptiveMaxPool1d)
AdaptiveMaxPool2d = wrap_module(nn.AdaptiveMaxPool2d)
AdaptiveMaxPool3d = wrap_module(nn.AdaptiveMaxPool3d)
AdaptiveAvgPool1d = wrap_module(nn.AdaptiveAvgPool1d)
AdaptiveAvgPool2d = wrap_module(nn.AdaptiveAvgPool2d)
AdaptiveAvgPool3d = wrap_module(nn.AdaptiveAvgPool3d)
TripletMarginLoss = wrap_module(nn.TripletMarginLoss)
ZeroPad2d = wrap_module(nn.ZeroPad2d)
ConstantPad1d = wrap_module(nn.ConstantPad1d)
ConstantPad2d = wrap_module(nn.ConstantPad2d)
ConstantPad3d = wrap_module(nn.ConstantPad3d)
Bilinear = wrap_module(nn.Bilinear)
CosineSimilarity = wrap_module(nn.CosineSimilarity)
Unfold = wrap_module(nn.Unfold)
Fold = wrap_module(nn.Fold)
AdaptiveLogSoftmaxWithLoss = wrap_module(nn.AdaptiveLogSoftmaxWithLoss)
TransformerEncoder = wrap_module(nn.TransformerEncoder)
TransformerDecoder = wrap_module(nn.TransformerDecoder)
TransformerEncoderLayer = wrap_module(nn.TransformerEncoderLayer)
TransformerDecoderLayer = wrap_module(nn.TransformerDecoderLayer)
Transformer = wrap_module(nn.Transformer)
#LazyLinear = wrap_module(nn.LazyLinear)
#LazyConv1d = wrap_module(nn.LazyConv1d)
#LazyConv2d = wrap_module(nn.LazyConv2d)
#LazyConv3d = wrap_module(nn.LazyConv3d)
#LazyConvTranspose1d = wrap_module(nn.LazyConvTranspose1d)
#LazyConvTranspose2d = wrap_module(nn.LazyConvTranspose2d)
#LazyConvTranspose3d = wrap_module(nn.LazyConvTranspose3d)
Flatten = wrap_module(nn.Flatten)
#Unflatten = wrap_module(nn.Unflatten)
Hardsigmoid = wrap_module(nn.Hardsigmoid)
Hardswish = wrap_module(nn.Hardswish)
#SiLU = wrap_module(nn.SiLU)
#TripletMarginWithDistanceLoss = wrap_module(nn.TripletMarginWithDistanceLoss)
#ChannelShuffle = wrap_module(nn.ChannelShuffle)
