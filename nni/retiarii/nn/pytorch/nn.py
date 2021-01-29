import logging
from typing import Any, List

import torch
import torch.nn as nn

from ...utils import add_record, blackbox_module, uid, version_larger_equal

# NOTE: support pytorch version >= 1.5.0

__all__ = [
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
    'Flatten', 'Hardsigmoid'
]

if version_larger_equal(torch.__version__, '1.6.0'):
    __all__.append('Hardswish')

if version_larger_equal(torch.__version__, '1.7.0'):
    __all__.extend(['Unflatten', 'SiLU', 'TripletMarginWithDistanceLoss'])


Module = nn.Module


class Sequential(nn.Sequential):
    def __init__(self, *args):
        add_record(id(self), {})
        super(Sequential, self).__init__(*args)


class ModuleList(nn.ModuleList):
    def __init__(self, *args):
        add_record(id(self), {})
        super(ModuleList, self).__init__(*args)


Identity = blackbox_module(nn.Identity)
Linear = blackbox_module(nn.Linear)
Conv1d = blackbox_module(nn.Conv1d)
Conv2d = blackbox_module(nn.Conv2d)
Conv3d = blackbox_module(nn.Conv3d)
ConvTranspose1d = blackbox_module(nn.ConvTranspose1d)
ConvTranspose2d = blackbox_module(nn.ConvTranspose2d)
ConvTranspose3d = blackbox_module(nn.ConvTranspose3d)
Threshold = blackbox_module(nn.Threshold)
ReLU = blackbox_module(nn.ReLU)
Hardtanh = blackbox_module(nn.Hardtanh)
ReLU6 = blackbox_module(nn.ReLU6)
Sigmoid = blackbox_module(nn.Sigmoid)
Tanh = blackbox_module(nn.Tanh)
Softmax = blackbox_module(nn.Softmax)
Softmax2d = blackbox_module(nn.Softmax2d)
LogSoftmax = blackbox_module(nn.LogSoftmax)
ELU = blackbox_module(nn.ELU)
SELU = blackbox_module(nn.SELU)
CELU = blackbox_module(nn.CELU)
GLU = blackbox_module(nn.GLU)
GELU = blackbox_module(nn.GELU)
Hardshrink = blackbox_module(nn.Hardshrink)
LeakyReLU = blackbox_module(nn.LeakyReLU)
LogSigmoid = blackbox_module(nn.LogSigmoid)
Softplus = blackbox_module(nn.Softplus)
Softshrink = blackbox_module(nn.Softshrink)
MultiheadAttention = blackbox_module(nn.MultiheadAttention)
PReLU = blackbox_module(nn.PReLU)
Softsign = blackbox_module(nn.Softsign)
Softmin = blackbox_module(nn.Softmin)
Tanhshrink = blackbox_module(nn.Tanhshrink)
RReLU = blackbox_module(nn.RReLU)
AvgPool1d = blackbox_module(nn.AvgPool1d)
AvgPool2d = blackbox_module(nn.AvgPool2d)
AvgPool3d = blackbox_module(nn.AvgPool3d)
MaxPool1d = blackbox_module(nn.MaxPool1d)
MaxPool2d = blackbox_module(nn.MaxPool2d)
MaxPool3d = blackbox_module(nn.MaxPool3d)
MaxUnpool1d = blackbox_module(nn.MaxUnpool1d)
MaxUnpool2d = blackbox_module(nn.MaxUnpool2d)
MaxUnpool3d = blackbox_module(nn.MaxUnpool3d)
FractionalMaxPool2d = blackbox_module(nn.FractionalMaxPool2d)
FractionalMaxPool3d = blackbox_module(nn.FractionalMaxPool3d)
LPPool1d = blackbox_module(nn.LPPool1d)
LPPool2d = blackbox_module(nn.LPPool2d)
LocalResponseNorm = blackbox_module(nn.LocalResponseNorm)
BatchNorm1d = blackbox_module(nn.BatchNorm1d)
BatchNorm2d = blackbox_module(nn.BatchNorm2d)
BatchNorm3d = blackbox_module(nn.BatchNorm3d)
InstanceNorm1d = blackbox_module(nn.InstanceNorm1d)
InstanceNorm2d = blackbox_module(nn.InstanceNorm2d)
InstanceNorm3d = blackbox_module(nn.InstanceNorm3d)
LayerNorm = blackbox_module(nn.LayerNorm)
GroupNorm = blackbox_module(nn.GroupNorm)
SyncBatchNorm = blackbox_module(nn.SyncBatchNorm)
Dropout = blackbox_module(nn.Dropout)
Dropout2d = blackbox_module(nn.Dropout2d)
Dropout3d = blackbox_module(nn.Dropout3d)
AlphaDropout = blackbox_module(nn.AlphaDropout)
FeatureAlphaDropout = blackbox_module(nn.FeatureAlphaDropout)
ReflectionPad1d = blackbox_module(nn.ReflectionPad1d)
ReflectionPad2d = blackbox_module(nn.ReflectionPad2d)
ReplicationPad2d = blackbox_module(nn.ReplicationPad2d)
ReplicationPad1d = blackbox_module(nn.ReplicationPad1d)
ReplicationPad3d = blackbox_module(nn.ReplicationPad3d)
CrossMapLRN2d = blackbox_module(nn.CrossMapLRN2d)
Embedding = blackbox_module(nn.Embedding)
EmbeddingBag = blackbox_module(nn.EmbeddingBag)
RNNBase = blackbox_module(nn.RNNBase)
RNN = blackbox_module(nn.RNN)
LSTM = blackbox_module(nn.LSTM)
GRU = blackbox_module(nn.GRU)
RNNCellBase = blackbox_module(nn.RNNCellBase)
RNNCell = blackbox_module(nn.RNNCell)
LSTMCell = blackbox_module(nn.LSTMCell)
GRUCell = blackbox_module(nn.GRUCell)
PixelShuffle = blackbox_module(nn.PixelShuffle)
Upsample = blackbox_module(nn.Upsample)
UpsamplingNearest2d = blackbox_module(nn.UpsamplingNearest2d)
UpsamplingBilinear2d = blackbox_module(nn.UpsamplingBilinear2d)
PairwiseDistance = blackbox_module(nn.PairwiseDistance)
AdaptiveMaxPool1d = blackbox_module(nn.AdaptiveMaxPool1d)
AdaptiveMaxPool2d = blackbox_module(nn.AdaptiveMaxPool2d)
AdaptiveMaxPool3d = blackbox_module(nn.AdaptiveMaxPool3d)
AdaptiveAvgPool1d = blackbox_module(nn.AdaptiveAvgPool1d)
AdaptiveAvgPool2d = blackbox_module(nn.AdaptiveAvgPool2d)
AdaptiveAvgPool3d = blackbox_module(nn.AdaptiveAvgPool3d)
TripletMarginLoss = blackbox_module(nn.TripletMarginLoss)
ZeroPad2d = blackbox_module(nn.ZeroPad2d)
ConstantPad1d = blackbox_module(nn.ConstantPad1d)
ConstantPad2d = blackbox_module(nn.ConstantPad2d)
ConstantPad3d = blackbox_module(nn.ConstantPad3d)
Bilinear = blackbox_module(nn.Bilinear)
CosineSimilarity = blackbox_module(nn.CosineSimilarity)
Unfold = blackbox_module(nn.Unfold)
Fold = blackbox_module(nn.Fold)
AdaptiveLogSoftmaxWithLoss = blackbox_module(nn.AdaptiveLogSoftmaxWithLoss)
TransformerEncoder = blackbox_module(nn.TransformerEncoder)
TransformerDecoder = blackbox_module(nn.TransformerDecoder)
TransformerEncoderLayer = blackbox_module(nn.TransformerEncoderLayer)
TransformerDecoderLayer = blackbox_module(nn.TransformerDecoderLayer)
Transformer = blackbox_module(nn.Transformer)
Flatten = blackbox_module(nn.Flatten)
Hardsigmoid = blackbox_module(nn.Hardsigmoid)

if version_larger_equal(torch.__version__, '1.6.0'):
    Hardswish = blackbox_module(nn.Hardswish)

if version_larger_equal(torch.__version__, '1.7.0'):
    SiLU = blackbox_module(nn.SiLU)
    Unflatten = blackbox_module(nn.Unflatten)
    TripletMarginWithDistanceLoss = blackbox_module(nn.TripletMarginWithDistanceLoss)
