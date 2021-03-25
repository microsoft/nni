# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from ...serializer import basic_unit
from ...serializer import transparent_serialize
from ...utils import version_larger_equal

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

Sequential = transparent_serialize(nn.Sequential)
ModuleList = transparent_serialize(nn.ModuleList)

Identity = basic_unit(nn.Identity)
Linear = basic_unit(nn.Linear)
Conv1d = basic_unit(nn.Conv1d)
Conv2d = basic_unit(nn.Conv2d)
Conv3d = basic_unit(nn.Conv3d)
ConvTranspose1d = basic_unit(nn.ConvTranspose1d)
ConvTranspose2d = basic_unit(nn.ConvTranspose2d)
ConvTranspose3d = basic_unit(nn.ConvTranspose3d)
Threshold = basic_unit(nn.Threshold)
ReLU = basic_unit(nn.ReLU)
Hardtanh = basic_unit(nn.Hardtanh)
ReLU6 = basic_unit(nn.ReLU6)
Sigmoid = basic_unit(nn.Sigmoid)
Tanh = basic_unit(nn.Tanh)
Softmax = basic_unit(nn.Softmax)
Softmax2d = basic_unit(nn.Softmax2d)
LogSoftmax = basic_unit(nn.LogSoftmax)
ELU = basic_unit(nn.ELU)
SELU = basic_unit(nn.SELU)
CELU = basic_unit(nn.CELU)
GLU = basic_unit(nn.GLU)
GELU = basic_unit(nn.GELU)
Hardshrink = basic_unit(nn.Hardshrink)
LeakyReLU = basic_unit(nn.LeakyReLU)
LogSigmoid = basic_unit(nn.LogSigmoid)
Softplus = basic_unit(nn.Softplus)
Softshrink = basic_unit(nn.Softshrink)
MultiheadAttention = basic_unit(nn.MultiheadAttention)
PReLU = basic_unit(nn.PReLU)
Softsign = basic_unit(nn.Softsign)
Softmin = basic_unit(nn.Softmin)
Tanhshrink = basic_unit(nn.Tanhshrink)
RReLU = basic_unit(nn.RReLU)
AvgPool1d = basic_unit(nn.AvgPool1d)
AvgPool2d = basic_unit(nn.AvgPool2d)
AvgPool3d = basic_unit(nn.AvgPool3d)
MaxPool1d = basic_unit(nn.MaxPool1d)
MaxPool2d = basic_unit(nn.MaxPool2d)
MaxPool3d = basic_unit(nn.MaxPool3d)
MaxUnpool1d = basic_unit(nn.MaxUnpool1d)
MaxUnpool2d = basic_unit(nn.MaxUnpool2d)
MaxUnpool3d = basic_unit(nn.MaxUnpool3d)
FractionalMaxPool2d = basic_unit(nn.FractionalMaxPool2d)
FractionalMaxPool3d = basic_unit(nn.FractionalMaxPool3d)
LPPool1d = basic_unit(nn.LPPool1d)
LPPool2d = basic_unit(nn.LPPool2d)
LocalResponseNorm = basic_unit(nn.LocalResponseNorm)
BatchNorm1d = basic_unit(nn.BatchNorm1d)
BatchNorm2d = basic_unit(nn.BatchNorm2d)
BatchNorm3d = basic_unit(nn.BatchNorm3d)
InstanceNorm1d = basic_unit(nn.InstanceNorm1d)
InstanceNorm2d = basic_unit(nn.InstanceNorm2d)
InstanceNorm3d = basic_unit(nn.InstanceNorm3d)
LayerNorm = basic_unit(nn.LayerNorm)
GroupNorm = basic_unit(nn.GroupNorm)
SyncBatchNorm = basic_unit(nn.SyncBatchNorm)
Dropout = basic_unit(nn.Dropout)
Dropout2d = basic_unit(nn.Dropout2d)
Dropout3d = basic_unit(nn.Dropout3d)
AlphaDropout = basic_unit(nn.AlphaDropout)
FeatureAlphaDropout = basic_unit(nn.FeatureAlphaDropout)
ReflectionPad1d = basic_unit(nn.ReflectionPad1d)
ReflectionPad2d = basic_unit(nn.ReflectionPad2d)
ReplicationPad2d = basic_unit(nn.ReplicationPad2d)
ReplicationPad1d = basic_unit(nn.ReplicationPad1d)
ReplicationPad3d = basic_unit(nn.ReplicationPad3d)
CrossMapLRN2d = basic_unit(nn.CrossMapLRN2d)
Embedding = basic_unit(nn.Embedding)
EmbeddingBag = basic_unit(nn.EmbeddingBag)
RNNBase = basic_unit(nn.RNNBase)
RNN = basic_unit(nn.RNN)
LSTM = basic_unit(nn.LSTM)
GRU = basic_unit(nn.GRU)
RNNCellBase = basic_unit(nn.RNNCellBase)
RNNCell = basic_unit(nn.RNNCell)
LSTMCell = basic_unit(nn.LSTMCell)
GRUCell = basic_unit(nn.GRUCell)
PixelShuffle = basic_unit(nn.PixelShuffle)
Upsample = basic_unit(nn.Upsample)
UpsamplingNearest2d = basic_unit(nn.UpsamplingNearest2d)
UpsamplingBilinear2d = basic_unit(nn.UpsamplingBilinear2d)
PairwiseDistance = basic_unit(nn.PairwiseDistance)
AdaptiveMaxPool1d = basic_unit(nn.AdaptiveMaxPool1d)
AdaptiveMaxPool2d = basic_unit(nn.AdaptiveMaxPool2d)
AdaptiveMaxPool3d = basic_unit(nn.AdaptiveMaxPool3d)
AdaptiveAvgPool1d = basic_unit(nn.AdaptiveAvgPool1d)
AdaptiveAvgPool2d = basic_unit(nn.AdaptiveAvgPool2d)
AdaptiveAvgPool3d = basic_unit(nn.AdaptiveAvgPool3d)
TripletMarginLoss = basic_unit(nn.TripletMarginLoss)
ZeroPad2d = basic_unit(nn.ZeroPad2d)
ConstantPad1d = basic_unit(nn.ConstantPad1d)
ConstantPad2d = basic_unit(nn.ConstantPad2d)
ConstantPad3d = basic_unit(nn.ConstantPad3d)
Bilinear = basic_unit(nn.Bilinear)
CosineSimilarity = basic_unit(nn.CosineSimilarity)
Unfold = basic_unit(nn.Unfold)
Fold = basic_unit(nn.Fold)
AdaptiveLogSoftmaxWithLoss = basic_unit(nn.AdaptiveLogSoftmaxWithLoss)
TransformerEncoder = basic_unit(nn.TransformerEncoder)
TransformerDecoder = basic_unit(nn.TransformerDecoder)
TransformerEncoderLayer = basic_unit(nn.TransformerEncoderLayer)
TransformerDecoderLayer = basic_unit(nn.TransformerDecoderLayer)
Transformer = basic_unit(nn.Transformer)
Flatten = basic_unit(nn.Flatten)
Hardsigmoid = basic_unit(nn.Hardsigmoid)

if version_larger_equal(torch.__version__, '1.6.0'):
    Hardswish = basic_unit(nn.Hardswish)

if version_larger_equal(torch.__version__, '1.7.0'):
    SiLU = basic_unit(nn.SiLU)
    Unflatten = basic_unit(nn.Unflatten)
    TripletMarginWithDistanceLoss = basic_unit(nn.TripletMarginWithDistanceLoss)
