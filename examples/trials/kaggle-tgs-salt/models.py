# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
from torchvision.models import resnet34, resnet101, resnet50, resnet152
import torchvision


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks
# https://arxiv.org/abs/1803.02579

class ChannelAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class SpatialAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.size())
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = ConvBn2d(in_channels, middle_channels)
        self.conv2 = ConvBn2d(middle_channels, out_channels)
        #self.deconv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x,e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = x*g1 + x*g2

        return x

class EncoderBlock(nn.Module):
    def __init__(self, block, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = block
        self.out_channels = out_channels
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x):
        x = self.block(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)

        return x*g1 + x*g2


def create_resnet(layers):
    if layers == 34:
        return resnet34(pretrained=True), 512
    elif layers == 50:
        return resnet50(pretrained=True), 2048
    elif layers == 101:
        return resnet101(pretrained=True), 2048
    elif layers == 152:
        return resnet152(pretrained=True), 2048
    else:
        raise NotImplementedError('only 34, 50, 101, 152 version of Resnet are implemented')

class UNetResNetV4(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.4,
                 pretrained=True, is_deconv=True):
        super(UNetResNetV4, self).__init__()
        self.name = 'UNetResNetV4_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.resnet, bottom_channel_nr = create_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            num_filters*2
        )
        self.encoder2 = EncoderBlock(self.resnet.layer1, bottom_channel_nr//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, bottom_channel_nr//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, bottom_channel_nr//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, bottom_channel_nr)

        center_block = nn.Sequential(
            ConvBn2d(bottom_channel_nr, bottom_channel_nr, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(bottom_channel_nr, bottom_channel_nr//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.center = EncoderBlock(center_block, bottom_channel_nr//2)

        self.decoder5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2,  num_filters * 16, 64)
        self.decoder4 = DecoderBlock(64 + bottom_channel_nr // 2,  num_filters * 8,  64)
        self.decoder3 = DecoderBlock(64 + bottom_channel_nr // 4,  num_filters * 4,  64)
        self.decoder2 = DecoderBlock(64 + bottom_channel_nr // 8, num_filters * 2,  64)
        self.decoder1 = DecoderBlock(64, num_filters, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.encoder1(x) #; print('x:', x.size())
        e2 = self.encoder2(x) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        d1 = self.decoder1(d2) #; print('d1:', d1.size())

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1)

        f = F.dropout2d(f, p=self.dropout_2d)

        return self.logit(f), None

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_params(self, base_lr):
        group1 = [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]
        group2 = [self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5, self.center, self.logit]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)

        param_group1 = {'params': params1, 'lr': base_lr / 5}

        params2 = []
        for x in group2:
            for p in x.parameters():
                params2.append(p)
        param_group2 = {'params': params2, 'lr': base_lr}

        return [param_group1, param_group2]

class DecoderBlockV5(nn.Module):
    def __init__(self, in_channels_x, in_channels_e, middle_channels, out_channels):
        super(DecoderBlockV5, self).__init__()
        self.in_channels = in_channels_x + in_channels_e
        self.conv1 = ConvBn2d(self.in_channels, middle_channels)
        self.conv2 = ConvBn2d(middle_channels, out_channels)
        self.deconv = nn.ConvTranspose2d(in_channels_x, in_channels_x, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x, e=None):
        #x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.deconv(x)
        if e is not None:
            x = torch.cat([x,e], 1)
        x = self.bn(x)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = x*g1 + x*g2

        return x



class UNetResNetV5(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.5):
        super(UNetResNetV5, self).__init__()
        self.name = 'UNetResNetV5_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.resnet, bottom_channel_nr = create_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            num_filters*2
        )
        self.encoder2 = EncoderBlock(self.resnet.layer1, bottom_channel_nr//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, bottom_channel_nr//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, bottom_channel_nr//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, bottom_channel_nr)

        center_block = nn.Sequential(
            ConvBn2d(bottom_channel_nr, bottom_channel_nr, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(bottom_channel_nr, bottom_channel_nr//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.center = EncoderBlock(center_block, bottom_channel_nr//2)

        self.decoder5 = DecoderBlockV5(bottom_channel_nr // 2, bottom_channel_nr,  num_filters * 16, 64)
        self.decoder4 = DecoderBlockV5(64, bottom_channel_nr // 2,  num_filters * 8,  64)
        self.decoder3 = DecoderBlockV5(64, bottom_channel_nr // 4,  num_filters * 4,  64)
        self.decoder2 = DecoderBlockV5(64, bottom_channel_nr // 8, num_filters * 2,  64)
        self.decoder1 = DecoderBlockV5(64, 0, num_filters, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.encoder1(x) #; print('x:', x.size())
        e2 = self.encoder2(x) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        d1 = self.decoder1(d2) #; print('d1:', d1.size())

        f = torch.cat([
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1)

        f = F.dropout2d(f, p=self.dropout_2d)

        return self.logit(f), None

class UNetResNetV6(nn.Module):
    '''
    1. Remove first pool from UNetResNetV5, such that resolution is doubled
    2. Remove scSE from center block
    3. Increase default dropout
    '''
    def __init__(self, encoder_depth, num_filters=32, dropout_2d=0.5):
        super(UNetResNetV6, self).__init__()
        assert encoder_depth == 34, 'UNetResNetV6: only 34 layers is supported!'
        self.name = 'UNetResNetV6_'+str(encoder_depth)
        self.dropout_2d = dropout_2d

        self.resnet, bottom_channel_nr = create_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            num_filters*2
        )

        self.encoder2 = EncoderBlock(self.resnet.layer1, bottom_channel_nr//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, bottom_channel_nr//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, bottom_channel_nr//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, bottom_channel_nr)

        self.center = nn.Sequential(
            ConvBn2d(bottom_channel_nr, bottom_channel_nr, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(bottom_channel_nr, bottom_channel_nr//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #self.center = EncoderBlock(center_block, bottom_channel_nr//2)

        self.decoder5 = DecoderBlockV5(bottom_channel_nr // 2, bottom_channel_nr,  num_filters * 16, 64)
        self.decoder4 = DecoderBlockV5(64, bottom_channel_nr // 2,  num_filters * 8,  64)
        self.decoder3 = DecoderBlockV5(64, bottom_channel_nr // 4,  num_filters * 4,  64)
        self.decoder2 = DecoderBlockV5(64, bottom_channel_nr // 8, num_filters * 2,  64)
        self.decoder1 = DecoderBlockV5(64, 0, num_filters, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.logit_image = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.encoder1(x) #; print('x:', x.size())
        e2 = self.encoder2(x) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        #d1 = self.decoder1(d2) ; print('d1:', d1.size())

        f = torch.cat([
            d2,
            F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(center, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1)

        f = F.dropout2d(f, p=self.dropout_2d, training=self.training)

        # empty mask classifier
        img_f = F.adaptive_avg_pool2d(e5, 1).view(x.size(0), -1)
        img_f = F.dropout(img_f, p=0.5, training=self.training)
        img_logit = self.logit_image(img_f).view(-1)

        return self.logit(f), img_logit


class DecoderBlockV7(nn.Module):
    def __init__(self, in_channels_x, in_channels_e, middle_channels, out_channels):
        super(DecoderBlockV7, self).__init__()
        self.in_channels = in_channels_x + in_channels_e
        self.conv1 = ConvBn2d(self.in_channels, middle_channels)
        self.conv2 = ConvBn2d(middle_channels, out_channels)
        self.deconv = nn.ConvTranspose2d(in_channels_x, in_channels_x, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x, e=None, upsample=True):
        #x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if upsample:
            x = self.deconv(x)
        if e is not None:
            x = torch.cat([x,e], 1)
        x = self.bn(x)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = x*g1 + x*g2

        return x

class UNet7(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.5):
        super(UNet7, self).__init__()
        nf = num_filters
        self.name = 'UNet7_'+str(encoder_depth)+'_nf'+str(nf)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.resnet, nbtm = create_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            64
        )
        self.encoder2 = EncoderBlock(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                self.resnet.layer1,
            ),
            nbtm//8
        )
        self.encoder3 = EncoderBlock(self.resnet.layer2, nbtm//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, nbtm//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, nbtm)

        center_block = nn.Sequential(
            ConvBn2d(nbtm, nbtm, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(nbtm, nbtm//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2) # remove
        )
        self.center = EncoderBlock(center_block, nbtm//2)

        self.decoder5 = DecoderBlockV7(nbtm // 2, nbtm,  nf * 16, nf*2)
        self.decoder4 = DecoderBlockV7(nf*2, nbtm // 2,  nf * 8,  nf*2)
        self.decoder3 = DecoderBlockV7(nf*2, nbtm // 4,  nf * 4,  nf*2)
        self.decoder2 = DecoderBlockV7(nf*2, nbtm // 8,  nf * 2,  nf*2)
        self.decoder1 = DecoderBlockV7(nf*2, 64, nf*2, nf*2)

        self.logit = nn.Sequential(
            nn.Conv2d(nf*10, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.logit_image = nn.Sequential(
            nn.Linear(nbtm, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        e1 = self.encoder1(x) #; print('e1:', e1.size())
        e2 = self.encoder2(e1) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5, upsample=False) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        d1 = self.decoder1(d2, e1) #; print('d1:', d1.size())

        f = torch.cat([
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1)

        f = F.dropout2d(f, p=self.dropout_2d)

        # empty mask classifier
        img_f = F.adaptive_avg_pool2d(e5, 1).view(x.size(0), -1)
        img_f = F.dropout(img_f, p=0.5, training=self.training)
        img_logit = self.logit_image(img_f).view(-1)

        return self.logit(f), img_logit


class UNet8(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.5):
        super(UNet8, self).__init__()
        nf = num_filters
        self.name = 'UNet8_'+str(encoder_depth)+'_nf'+str(nf)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.resnet, nbtm = create_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            64
        )

        self.encoder2 = EncoderBlock(self.resnet.layer1, nbtm//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, nbtm//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, nbtm//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, nbtm)

        center_block = nn.Sequential(
            ConvBn2d(nbtm, nbtm, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(nbtm, nbtm//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2) # remove
        )
        self.center = EncoderBlock(center_block, nbtm//2)

        self.decoder5 = DecoderBlockV7(nbtm // 2, nbtm,  nf * 16, nf*2)
        self.decoder4 = DecoderBlockV7(nf*2, nbtm // 2,  nf * 8,  nf*2)
        self.decoder3 = DecoderBlockV7(nf*2, nbtm // 4,  nf * 4,  nf*2)
        self.decoder2 = DecoderBlockV7(nf*2, nbtm // 8,  nf * 2,  nf*2)
        self.decoder1 = DecoderBlockV7(nf*2+64, 3, nf*2, nf*2)

        self.logit = nn.Sequential(
            nn.Conv2d(nf*10, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.logit_image = nn.Sequential(
            nn.Linear(nbtm, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        e1 = self.encoder1(x) #; print('e1:', e1.size())
        e2 = self.encoder2(e1) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5, upsample=False) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        d1 = self.decoder1(torch.cat([d2, e1], 1), x) #; print('d1:', d1.size())

        f = torch.cat([
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1)

        f = F.dropout2d(f, p=self.dropout_2d)

        # empty mask classifier
        img_f = F.adaptive_avg_pool2d(e5, 1).view(x.size(0), -1)
        img_f = F.dropout(img_f, p=0.5, training=self.training)
        img_logit = self.logit_image(img_f).view(-1)

        return self.logit(f), img_logit


def test():
    model = UNet8(50, num_filters=32).cuda()
    inputs = torch.randn(2,3,128,128).cuda()
    out, _ = model(inputs)
    #print(model)
    print(out.size(), _.size()) #, cls_taret.size())
    #print(out)


if __name__ == '__main__':
    test()
